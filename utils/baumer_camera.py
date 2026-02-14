# utils/baumer_camera.py
"""
BaumerCamera - neoAPI Python バインディング最小ラッパー

機能:
- 単発撮影（grab_nonblock）
- 連続撮影（continuous_capture）
- 輝度トリガーモード（start_trigger_mode / stop_trigger_mode）
"""

from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import neoapi


@dataclass
class TriggerEvent:
    """トリガーイベント情報"""
    timestamp: float           # 検知時刻
    change_ratio: float        # 変化ピクセルの割合（0.0-1.0）
    frame: np.ndarray          # 検査対象フレーム（遅延後）


# トリガー状態定数
TRIGGER_STATE_WAITING = "waiting"      # 待機中（トリガー検知可能）
TRIGGER_STATE_DISABLED = "disabled"    # 無効中（トリガー検知不可）


class BaumerCamera:
    """
    neoAPI Python バインディング最小ラッパ

    機能:
      - grab_nonblock(): ノンブロッキングで1枚取得
      - continuous_capture(): 連続撮影で指定枚数取得
      - start_trigger_mode(): 輝度変化トリガーモード開始
      - stop_trigger_mode(): トリガーモード停止
    """

    # デフォルトのトリガー設定（フレーム間差分方式）
    DEFAULT_TRIGGER_CONFIG = {
        "detection_roi_rel": [0.0, 1.0, 0.0, 0.10],   # 検知ROI（相対座標）
        "diff_threshold": 25,                          # 差分しきい値（0-255）
        "change_ratio_threshold": 0.05,                # 変化割合しきい値（0.0-1.0）
        "capture_delay": 0.3,                          # 検知後の撮像遅延（秒）
        "trigger_interval": 2.0,                       # トリガー発動後の無効時間（秒）
    }

    def __init__(self, *, step_down: int = 10) -> None:
        self.cam = neoapi.Cam()
        self._step_down = step_down

        # デバイス検出とエラーハンドリング
        try:
            device_list = neoapi.CamInfoList_Get()
            device_list.Refresh()
            if device_list.size() == 0:
                raise RuntimeError("カメラが見つかりません。カメラが接続されているか確認してください。")
            elif device_list.size() > 1:
                print(f"警告: {device_list.size()}台のカメラが検出されました。最初のカメラを使用します。")

            # 最初のデバイスに接続
            cam_info = device_list[0]
            self.cam.Connect(cam_info.GetId())
            print(f"カメラに接続しました: {cam_info.GetModelName()}")

        except neoapi.NeoException as e:
            raise RuntimeError(f"カメラ接続エラー: {e}")

        if self.cam.HasFeature("TriggerMode"):      # FreeRun へ
            self.cam.f.TriggerMode.SetString("Off")

        # トリガーモード関連の初期化
        self._trigger_config: Dict = self.DEFAULT_TRIGGER_CONFIG.copy()
        self._trigger_running: bool = False
        self._trigger_thread: Optional[threading.Thread] = None
        self._prev_frame: Optional[np.ndarray] = None      # 前フレーム（差分計算用）
        self._last_trigger_time: float = 0.0               # 最後のトリガー発動時刻
        self._is_disabled: bool = False                    # 無効状態フラグ
        self._frame_buffer: deque = deque(maxlen=100)
        self._on_trigger: Optional[Callable[[TriggerEvent], None]] = None
        self._on_frame: Optional[Callable[[np.ndarray], None]] = None
        self._on_state_change: Optional[Callable[[str, float], None]] = None  # (状態, 残り時間)
        self._trigger_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _alloc_buffers(self, desired: int) -> None:
        """リングバッファ確保。失敗したら step_down 枚ずつ減らす"""
        count = max(desired, 10)
        while count > 0:
            try:
                self.cam.SetImageBufferCount(count)
                self.cam.SetImageBufferCycleCount(count)
                return
            except neoapi.NeoException:
                count -= self._step_down
        raise RuntimeError("バッファを確保できませんでした。")

    # ------------------------------------------------------------------ #
    def apply_config(self, params: Dict[str, float | int | str | bool]) -> None:
        """camera_params.json を高速適用（最適化済み）"""
        
        # 優先順位付きパラメータ設定（重要なもの順）
        PRIORITY_ORDER = [
            "TriggerMode", "AcquisitionMode", "ExposureMode",           # 基本設定
            "TriggerSource", "TriggerSelector", "TriggerActivation",    # トリガー設定
            "Width", "Height", "OffsetX", "OffsetY",                   # 撮影範囲
            "ExposureTime", "Gain", "ExposureAuto", "GainAuto",        # 画質
            "AcquisitionFrameRateEnable", "AcquisitionFrameRate",      # フレームレート
            "PixelFormat",                                             # フォーマット
            "UserSetSelector", "UserSetDefault"                        # ユーザー設定
        ]

        # フレームレート有効化（高速化: try/except簡略化）
        try:
            if self.cam.HasFeature("AcquisitionFrameRateEnable"):
                self.cam.f.AcquisitionFrameRateEnable.value = True
        except:
            pass

        # 1. 優先パラメータを先に設定
        for param_name in PRIORITY_ORDER:
            if param_name in params:
                self._set_single_param_fast(param_name, params[param_name])

        # 2. 残りのパラメータを設定
        for k, v in params.items():
            if k in PRIORITY_ORDER:
                continue  # 既に設定済み
            self._set_single_param_fast(k, v)

        print(f"高速カメラ設定完了: {len(params)}個のパラメータを適用")

    def _set_single_param_fast(self, k: str, v) -> None:
        """単一パラメータ設定（最高速版）"""
        try:
            # HasFeatureチェックなしで高速化（存在しない場合はAttributeErrorで例外処理）
            feat = getattr(self.cam.f, k)
            
            # IsWritableチェックも簡略化
            if hasattr(feat, 'IsWritable') and not feat.IsWritable():
                return
            
            # 型別高速処理
            if isinstance(v, bool):
                # bool型: 1/0に変換してvalue設定
                if hasattr(feat, "value"):
                    feat.value = 1 if v else 0
                elif hasattr(feat, "SetString"):
                    feat.SetString("1" if v else "0")
            elif isinstance(v, (int, float)):
                # 数値型: 直接value設定
                if hasattr(feat, "value"):
                    feat.value = v
                else:
                    feat.SetString(str(v))
            else:
                # 文字列型: 特殊パラメータのみ判定
                if k in {"ActionGroupMask", "ActionGroupKey"} or k.startswith("Gev"):
                    # 16進数・IPアドレス系は文字列として設定
                    if hasattr(feat, "SetString"):
                        feat.SetString(str(v))
                elif hasattr(feat, "SetString"):
                    feat.SetString(v)
                elif hasattr(feat, "value"):
                    # 文字列から数値への変換を試行
                    try:
                        feat.value = float(v) if '.' in str(v) else int(v)
                    except (ValueError, TypeError):
                        feat.value = v
                    
        except AttributeError:
            # パラメータが存在しない（HasFeature相当）
            pass
        except Exception as e:
            # その他のエラーは警告のみ（処理続行）
            print(f"パラメータ設定警告 {k}={v}: {e}")

    # ------------------------------------------------------------------ #
    def grab_nonblock(self) -> np.ndarray | None:
        """ノンブロッキングで 1 枚取得（無ければ None）"""
        buf = self.cam.GetImage(0)
        if buf.IsEmpty():
            return None

        fmt = self.cam.f.PixelFormat.GetString()
        img = (
            buf.Convert("BGR8").GetNPArray()
            if "Bayer" in fmt
            else (
                cv2.cvtColor(buf.GetNPArray(), cv2.COLOR_RGB2BGR)
                if fmt == "RGB8"
                else buf.GetNPArray()
            )
        )
        # BGR to RGB conversion for downstream processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb.copy()

    # ------------------------------------------------------------------ #
    def continuous_capture(self, count: int, max_wait_time: float = 20.0) -> List[np.ndarray]:
        """
        連続撮影で指定枚数の画像を取得
        count: 取得したい画像数
        max_wait_time: 最大待機時間（秒）
        """
        try:
            # フレームレート取得
            fps = (
                self.cam.f.AcquisitionFrameRate.value
                if self.cam.HasFeature("AcquisitionFrameRate")
                else 60
            )
            
            # バッファを十分確保（要求枚数 + マージン）
            self._alloc_buffers(count + 20)
            
            frames: List[np.ndarray] = []
            
            # ストリーミング開始
            self.cam.StartStreaming()
            print(f"ストリーミング開始（目標: {count}枚, FPS: {fps}）")
            
            start_time = time.time()
            timeout_per_frame = max(1.0 / fps * 2, 0.1)  # フレーム間隔の2倍、最低0.1秒
            
            while len(frames) < count:
                current_time = time.time()
                
                # 全体のタイムアウトチェック
                if current_time - start_time > max_wait_time:
                    print(f"タイムアウト: {len(frames)}枚取得後に中断")
                    break
                
                # 画像取得を試行
                frame_start = time.time()
                while time.time() - frame_start < timeout_per_frame:
                    img = self.grab_nonblock()
                    if img is not None:
                        frames.append(img)
                        if len(frames) % 10 == 0:  # 10枚ごとに進捗表示
                            print(f"進捗: {len(frames)}/{count}枚取得")
                        break
                    time.sleep(0.001)  # 1ms待機
                else:
                    # フレーム取得タイムアウト
                    print(f"フレーム取得タイムアウト（{len(frames)}枚取得済み）")
                    
            self.cam.StopStreaming()
            print(f"撮影完了: {len(frames)}枚取得")
            return frames
            
        except Exception as e:
            print(f"連続撮影エラー: {e}")
            try:
                if self.cam.IsStreaming():
                    self.cam.StopStreaming()
            except:
                pass
            return frames if 'frames' in locals() else []

    # ------------------------------------------------------------------ #
    # トリガーモード関連
    # ------------------------------------------------------------------ #
    def set_trigger_config(self, config: Dict) -> None:
        """
        トリガー設定を適用

        Args:
            config: トリガー設定（system_params.jsonのtriggerセクション）
        """
        with self._trigger_lock:
            self._trigger_config = {**self.DEFAULT_TRIGGER_CONFIG, **config}

    def get_trigger_config(self) -> Dict:
        """現在のトリガー設定を取得"""
        with self._trigger_lock:
            return self._trigger_config.copy()

    @property
    def is_trigger_running(self) -> bool:
        """トリガーモードが実行中かどうか"""
        return self._trigger_running

    def start_trigger_mode(
        self,
        on_trigger: Callable[[TriggerEvent], None],
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
        on_state_change: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """
        輝度変化トリガーモードを開始

        Args:
            on_trigger: トリガー検知時のコールバック（TriggerEventを受け取る）
            on_frame: フレーム取得時のコールバック（プレビュー用、省略可）
            on_state_change: 状態変化時のコールバック（状態, 残り時間）

        Returns:
            開始成功時True
        """
        if self._trigger_running:
            print("トリガーモードは既に実行中です")
            return False

        self._on_trigger = on_trigger
        self._on_frame = on_frame
        self._on_state_change = on_state_change
        self._trigger_running = True
        self._prev_frame = None
        self._last_trigger_time = 0.0
        self._is_disabled = False
        self._frame_buffer.clear()

        self._trigger_thread = threading.Thread(
            target=self._trigger_loop,
            daemon=True,
        )
        self._trigger_thread.start()
        print("トリガーモードを開始しました")
        return True

    def stop_trigger_mode(self) -> None:
        """トリガーモードを停止"""
        if not self._trigger_running:
            return

        self._trigger_running = False
        if self._trigger_thread is not None:
            self._trigger_thread.join(timeout=3.0)
            self._trigger_thread = None

        self._on_trigger = None
        self._on_frame = None
        self._on_state_change = None
        print("トリガーモードを停止しました")

    def _trigger_loop(self) -> None:
        """トリガー監視ループ（別スレッドで実行）"""
        try:
            # バッファ確保とストリーミング開始
            self._alloc_buffers(50)
            self.cam.StartStreaming()
            print("トリガー監視: ストリーミング開始")

            while self._trigger_running:
                # フレーム取得
                frame = self.grab_nonblock()
                if frame is None:
                    time.sleep(0.001)
                    continue

                current_time = time.time()

                # フレームバッファに追加（遅延取得用）
                self._frame_buffer.append((current_time, frame))

                # プレビュー用コールバック
                if self._on_frame is not None:
                    try:
                        self._on_frame(frame)
                    except Exception as e:
                        print(f"プレビューコールバックエラー: {e}")

                # トリガー判定
                event = self._check_trigger(frame, current_time)
                if event is not None and self._on_trigger is not None:
                    try:
                        self._on_trigger(event)
                    except Exception as e:
                        print(f"トリガーコールバックエラー: {e}")

        except Exception as e:
            print(f"トリガーループエラー: {e}")
        finally:
            # ストリーミング停止
            try:
                if self.cam.IsStreaming():
                    self.cam.StopStreaming()
                print("トリガー監視: ストリーミング停止")
            except Exception as e:
                print(f"ストリーミング停止エラー: {e}")

    def _check_trigger(self, frame: np.ndarray, current_time: float) -> Optional[TriggerEvent]:
        """
        トリガー発火判定（フレーム間差分 + 一定時間無効化方式）

        Args:
            frame: 現在のフレーム
            current_time: 現在時刻

        Returns:
            トリガー発火時はTriggerEvent、それ以外はNone
        """
        with self._trigger_lock:
            config = self._trigger_config

        trigger_interval = config.get("trigger_interval", 2.0)
        elapsed = current_time - self._last_trigger_time
        remaining = trigger_interval - elapsed

        # 無効時間中の処理
        if remaining > 0:
            if not self._is_disabled:
                self._is_disabled = True
            # 毎フレーム残り時間を通知（UI更新用）
            self._notify_state_change(TRIGGER_STATE_DISABLED, remaining)
            return None

        # 無効時間終了時の処理
        if self._is_disabled:
            self._is_disabled = False
            self._prev_frame = None  # フレームリセット（復帰直後の誤発火防止）
            self._notify_state_change(TRIGGER_STATE_WAITING, 0.0)
            print("トリガー復帰: 待機中")

        # ROI切り出し
        roi = self._crop_roi(frame, config["detection_roi_rel"])
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # 前フレームがない場合は初期化（復帰直後は1フレームスキップ）
        if self._prev_frame is None:
            self._prev_frame = roi_gray.copy()
            return None

        # フレーム間差分を計算
        diff = cv2.absdiff(roi_gray, self._prev_frame)
        diff_threshold = config.get("diff_threshold", 25)
        _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # 変化ピクセルの割合を計算
        total_pixels = thresh.size
        changed_pixels = np.count_nonzero(thresh)
        change_ratio = changed_pixels / total_pixels

        # 前フレームを更新
        self._prev_frame = roi_gray.copy()

        # 変化割合がしきい値以上ならトリガー発火
        ratio_threshold = config.get("change_ratio_threshold", 0.05)
        if change_ratio >= ratio_threshold:
            self._last_trigger_time = current_time
            print(f"トリガー検知: 変化率={change_ratio:.3f}")

            # 遅延後のフレームを取得
            delay = config.get("capture_delay", 0.3)
            inspection_frame = self._get_delayed_frame(current_time, delay)

            return TriggerEvent(
                timestamp=current_time,
                change_ratio=change_ratio,
                frame=inspection_frame,
            )

        return None

    def _notify_state_change(self, state: str, remaining: float) -> None:
        """状態変化を通知"""
        if self._on_state_change is not None:
            try:
                self._on_state_change(state, remaining)
            except Exception as e:
                print(f"状態変化コールバックエラー: {e}")

    def _crop_roi(self, frame: np.ndarray, roi_rel: List[float]) -> np.ndarray:
        """
        ROI領域を切り出し

        Args:
            frame: 入力フレーム
            roi_rel: 相対座標 [y_start, y_end, x_start, x_end]

        Returns:
            切り出したROI領域
        """
        h, w = frame.shape[:2]
        y0 = int(roi_rel[0] * h)
        y1 = int(roi_rel[1] * h)
        x0 = int(roi_rel[2] * w)
        x1 = int(roi_rel[3] * w)
        return frame[y0:y1, x0:x1]

    def _calculate_brightness(self, roi: np.ndarray) -> float:
        """
        ROI領域の平均輝度を計算

        Args:
            roi: ROI領域（RGB画像）

        Returns:
            平均輝度値（0-255）
        """
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        return float(np.mean(gray))

    def _get_delayed_frame(self, trigger_time: float, delay: float) -> np.ndarray:
        """
        遅延後のフレームを取得

        Args:
            trigger_time: トリガー発火時刻
            delay: 遅延時間（秒）

        Returns:
            遅延後のフレーム。取得できない場合は最新フレーム
        """
        target_time = trigger_time + delay

        # 遅延時間まで待機しながらフレームを収集
        while time.time() < target_time and self._trigger_running:
            frame = self.grab_nonblock()
            if frame is not None:
                self._frame_buffer.append((time.time(), frame))
            time.sleep(0.001)

        # バッファから最も近い時刻のフレームを探す
        best_frame = None
        best_diff = float("inf")

        for timestamp, frame in self._frame_buffer:
            if timestamp >= target_time:
                diff = timestamp - target_time
                if diff < best_diff:
                    best_diff = diff
                    best_frame = frame

        # 見つからない場合は最新のフレームを返す
        if best_frame is None and self._frame_buffer:
            _, best_frame = self._frame_buffer[-1]

        return best_frame if best_frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    def __enter__(self): return self

    def __exit__(self, *_):
        # トリガーモード停止
        if self._trigger_running:
            self.stop_trigger_mode()

        try:
            if self.cam.IsStreaming():
                self.cam.StopStreaming()
        except Exception as e:
            print(f"ストリーミング停止エラー: {e}")

        try:
            self.cam.Disconnect()
        except Exception as e:
            print(f"カメラ切断エラー: {e}")