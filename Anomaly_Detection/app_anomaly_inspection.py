#!/usr/bin/env python3
# Anomaly_Detection/app_anomaly_inspection.py
"""
Baumer 産業カメラ × AnomalyDINO
フィルム取り違え検査アプリ（TkEasyGUI 版）

機能:
- 品種選択
- マスター画像との比較検査
- OK/NG判定
- パラメータ設定
"""

from __future__ import annotations
import datetime as _dt
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps
import cv2
import numpy as np
import TkEasyGUI as eg
import commentjson as cj

# プロジェクトルートをパスに追加
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.anomaly_dino_detector import AnomalyDINODetector
from utils.baumer_camera import (
    BaumerCamera,
    TriggerEvent,
    TRIGGER_STATE_WAITING,
    TRIGGER_STATE_DISABLED,
)


# =============================================================================
# ヘルパー関数
# =============================================================================
def imread_jp(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """
    日本語パスに対応した画像読み込み関数
    OpenCVのimreadは日本語パスを扱えないため、numpy経由で読み込む
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def maximize_window_safe(win: eg.Window, min_size: Tuple[int, int]) -> None:
    """OS対応のウィンドウ最大化処理"""
    tk_root = getattr(win, "window", None) or getattr(win, "TKroot", None)
    if tk_root:
        tk_root.attributes("-zoomed", True)


def play_buzzer_sound():
    """ブザー音を再生"""
    buzzer_path = _project_root / "Buzzer.wav"
    if buzzer_path.exists():
        subprocess.Popen(
            ["paplay", str(buzzer_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def pad_to_square(img: np.ndarray, size: int) -> np.ndarray:
    """
    画像を正方形にパディング・リサイズ

    Args:
        img: 入力画像（RGB形式を想定）
        size: 出力画像のサイズ（正方形）

    Returns:
        正方形にパディング・リサイズされた画像（RGB形式）
    """
    # 入力画像はBaumerCameraクラスでRGBに変換済み
    pil = Image.fromarray(img)  # RGB形式の画像をそのままPILに変換
    pil = ImageOps.pad(pil, (size, size), color=(0, 0, 0))
    return np.asarray(pil)  # RGB形式のまま返却


# =============================================================================
# AnomalyInspectionApp クラス（異常検知モード）
# =============================================================================
class AnomalyInspectionApp:
    """Baumer産業カメラ × AnomalyDINO 検査アプリ"""

    # -------------------------------------------------------------------------
    # 初期化
    # -------------------------------------------------------------------------
    def __init__(self, root: Path):
        self.root = root
        self.cfg = root / "config"  # 共有設定（カメラ、システム）
        self.anomaly_cfg = root / "Anomaly_Detection" / "config"  # 異常検知専用設定
        self.master_dir = root / "Anomaly_Detection" / "master"
        self.log = root / "Anomaly_Detection" / "logs"
        self.log.mkdir(exist_ok=True)

        # 設定ファイル読み込み
        self.base_cam = self._load_json("base_camera_params.json")  # 共有
        self.base_model = self._load_json("base_model_params.json", use_anomaly_cfg=True)  # 異常検知専用
        self.sys = self._load_json("system_params.json")  # 共有

        # システム設定
        self.buf_step = int(self.sys.get("buffer_step_down", 10))
        self.display_px = int(self.sys.get("display_size", 640))
        self.win_size = tuple(self.sys.get("window_size", [1600, 900]))

        # 検査関連
        self.detector: AnomalyDINODetector | None = None
        self.master_img: np.ndarray | None = None
        self.master_padded: np.ndarray | None = None
        self.cam_params: Dict[str, float | int | str] = {}
        self.model_params: Dict[str, float | int | str] = {}

        # カメラ関連
        self.camera: BaumerCamera | None = None
        self.camera_lock = threading.Lock()

        # 撮影処理中フラグ
        self.is_capturing = False

        # トリガーモード関連
        self.trigger_mode: bool = False
        self.preview_frame: np.ndarray | None = None
        self.pending_inspection: bool = False
        self.inspection_frame: np.ndarray | None = None
        self.trigger_state: str = TRIGGER_STATE_WAITING  # トリガー状態
        self.trigger_remaining: float = 0.0              # 無効時間残り（秒）

        eg.set_theme("clam")

    # -------------------------------------------------------------------------
    # JSON設定ファイル読み込み
    # -------------------------------------------------------------------------
    def _load_json(self, target: str | Path, use_anomaly_cfg: bool = False) -> Dict:
        """
        設定ファイル読み込み（存在しない場合は空辞書を返す）

        Args:
            target: ファイル名またはフルパス
            use_anomaly_cfg: Trueの場合、異常検知専用設定フォルダから読み込み
        """
        if isinstance(target, str):
            cfg_dir = self.anomaly_cfg if use_anomaly_cfg else self.cfg
            path = cfg_dir / target
        else:
            path = target
        if not path or not path.exists():
            return {}
        return cj.loads(path.read_text(encoding="utf-8"))

    # -------------------------------------------------------------------------
    # 製品設定ロード
    # -------------------------------------------------------------------------
    def list_products(self) -> List[str]:
        return [p.name for p in self.master_dir.iterdir() if p.is_dir()]

    def _initialize_camera(self) -> bool:
        """カメラ初期化・設定適用"""
        with self.camera_lock:
            # 既存カメラを切断
            if self.camera is not None:
                if self.camera.cam.IsStreaming():
                    self.camera.cam.StopStreaming()
                self.camera.cam.Disconnect()
                self.camera = None

            # 新しいカメラインスタンス作成と設定適用
            self.camera = BaumerCamera(step_down=self.buf_step)
            self.camera.apply_config(self.cam_params)
            print("カメラパラメータを適用しました")
            return True

    def _cleanup_camera(self):
        """カメラのクリーンアップ"""
        with self.camera_lock:
            if self.camera is not None:
                if self.camera.cam.IsStreaming():
                    self.camera.cam.StopStreaming()
                self.camera.cam.Disconnect()
                self.camera = None

    def load_product(self, prod: str):
        """
        指定製品の設定とマスター画像を読み込み
        """
        pdir = self.master_dir / prod

        # master.bmpがあれば使用、なければ最初の.bmpファイルを使用
        master_bmp_path = pdir / "master.bmp"
        if not master_bmp_path.exists():
            bmp_files = sorted(pdir.glob("*.bmp"))
            if bmp_files:
                master_bmp_path = bmp_files[0]
            else:
                raise FileNotFoundError(f"マスター画像が見つかりません: {pdir}")

        master_bgr = imread_jp(master_bmp_path)
        self.master_img = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2RGB)
        self.master_padded = pad_to_square(self.master_img, self.display_px)

        # 設定ファイル読み込み
        self.cam_params = self.base_cam.copy()
        if (pdir / "camera_params.json").exists():
            self.cam_params.update(self._load_json(pdir / "camera_params.json"))

        self.model_params = self.base_model.copy()
        if (pdir / "model_params.json").exists():
            self.model_params.update(self._load_json(pdir / "model_params.json"))

        # 検査エンジン初期化
        master_feat_path = pdir / "master.npy"
        self.detector = AnomalyDINODetector(
            self.model_params["model_type"],
            int(self.model_params["feat_layer"]),
            master_feat_path,
            int(self.model_params["image_size"]),
            threshold=float(self.model_params["threshold"]),
            roi_rel=tuple(self.model_params["roi_rel"]),
            sat_thresh=int(self.model_params["sat_thresh"]),
            hue_weight=float(self.model_params["hue_weight"]),
            product_name=prod,
        )

    # -------------------------------------------------------------------------
    # UI画面生成
    # -------------------------------------------------------------------------
    def _make_select_window(self) -> eg.Window:
        layout = [
            [eg.Text("検査品種を選択してください", font=("Arial", 22))],
            [eg.Listbox(self.list_products(), size=(60, 8), key="-LIST-")],
            [
                eg.Button("検査開始", key="-START-"),
                eg.Button("パラメータ設定", key="-SETTINGS-"),
                eg.Button("終了", key="-EXIT-"),
            ],
        ]
        win = eg.Window(
            "品種選択", layout, finalize=True, size=self.win_size, resizable=True
        )
        maximize_window_safe(win, self.win_size)
        return win

    def _make_settings_window(self) -> eg.Window:
        layout = [
            [eg.Text("システムパラメータ設定", font=("Arial", 22))],
            [
                eg.Text("バッファステップダウン:"),
                eg.InputText(
                    str(self.sys.get("buffer_step_down", 10)),
                    key="-BUFFER_STEP_DOWN-",
                    size=(10, 1),
                ),
            ],
            [
                eg.Text("表示サイズ:"),
                eg.InputText(
                    str(self.sys.get("display_size", 640)),
                    key="-DISPLAY_SIZE-",
                    size=(10, 1),
                ),
            ],
            [
                eg.Text("ウィンドウサイズ (幅):"),
                eg.InputText(
                    str(self.sys.get("window_size", [1600, 900])[0]),
                    key="-WINDOW_WIDTH-",
                    size=(10, 1),
                ),
            ],
            [
                eg.Text("ウィンドウサイズ (高さ):"),
                eg.InputText(
                    str(self.sys.get("window_size", [1600, 900])[1]),
                    key="-WINDOW_HEIGHT-",
                    size=(10, 1),
                ),
            ],
            [
                eg.Text("最大ログファイル数:"),
                eg.InputText(
                    str(self.sys.get("max_log_files", 1000)),
                    key="-MAX_LOG_FILES-",
                    size=(10, 1),
                ),
            ],
            [eg.HSeparator()],
            [eg.Text("しきい値設定", font=("Arial", 18))],
            [
                eg.Text("ベース（デフォルト）しきい値:"),
                eg.InputText(
                    str(self.base_model.get("threshold", 0.3)),
                    key="-BASE_THRESHOLD-",
                    size=(10, 1),
                ),
            ],
            [eg.Text("製品別しきい値設定:", font=("Arial", 14))],
        ]

        # 製品別しきい値設定を追加
        products = self.list_products()
        for prod in products:
            prod_threshold = self._get_product_threshold(prod)
            layout.append(
                [
                    eg.Text(f"{prod}:"),
                    eg.InputText(
                        str(prod_threshold), key=f"-THRESH_{prod}-", size=(10, 1)
                    ),
                    eg.Button("リセット", key=f"-RESET_{prod}-", size=(8, 1)),
                ]
            )

        layout.extend(
            [
                [eg.HSeparator()],
                [
                    eg.Button("保存", key="-SAVE_SETTINGS-"),
                    eg.Button("キャンセル", key="-CANCEL_SETTINGS-"),
                ],
            ]
        )

        win = eg.Window(
            "パラメータ設定", layout, finalize=True, size=(600, 800), resizable=True
        )
        return win

    def _load_product_params(self, prod: str) -> Dict:
        """製品固有設定を読み込み（ヘルパーメソッド）"""
        prod_model_path = self.master_dir / prod / "model_params.json"
        return self._load_json(prod_model_path) if prod_model_path.exists() else {}

    def _get_product_threshold(self, prod: str) -> float:
        """製品別しきい値取得"""
        prod_params = self._load_product_params(prod)
        return float(prod_params.get("threshold", self.base_model.get("threshold", 0.3)))

    def _save_product_threshold(
        self, prod: str, threshold: float, base_threshold: float
    ):
        """製品別しきい値保存"""
        prod_model_path = self.master_dir / prod / "model_params.json"

        # 既存設定読み込み
        prod_params = self._load_product_params(prod)

        # しきい値がベース値と同じなら設定から削除、異なるなら追加
        if abs(threshold - base_threshold) < 1e-6:
            # ベース値と同じなら設定を削除
            if "threshold" in prod_params:
                del prod_params["threshold"]
        else:
            # 異なるなら設定を追加
            prod_params["threshold"] = threshold

        # 設定を保存
        if prod_params:
            with open(prod_model_path, "w", encoding="utf-8") as f:
                json.dump(prod_params, f, indent=2, ensure_ascii=False)
        else:
            # 設定が空なら設定ファイルを削除
            if prod_model_path.exists():
                prod_model_path.unlink()

    def _save_settings(self, values: Dict) -> bool:
        try:
            new_params = {
                "buffer_step_down": int(values["-BUFFER_STEP_DOWN-"]),
                "display_size": int(values["-DISPLAY_SIZE-"]),
                "window_size": [
                    int(values["-WINDOW_WIDTH-"]),
                    int(values["-WINDOW_HEIGHT-"]),
                ],
                "max_log_files": int(values["-MAX_LOG_FILES-"]),
            }

            # システム設定を保存
            config_path = self.cfg / "system_params.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(new_params, f, indent=2, ensure_ascii=False)

            # ベースしきい値の保存
            base_threshold = float(values["-BASE_THRESHOLD-"])
            self.base_model["threshold"] = base_threshold
            base_model_path = self.cfg / "base_model_params.json"
            with open(base_model_path, "w", encoding="utf-8") as f:
                json.dump(self.base_model, f, indent=2, ensure_ascii=False)

            # 製品別しきい値の保存
            products = self.list_products()
            for prod in products:
                threshold_key = f"-THRESH_{prod}-"
                if threshold_key in values:
                    prod_threshold = float(values[threshold_key])
                    self._save_product_threshold(prod, prod_threshold, base_threshold)

            # 設定パラメータのログを保存
            self._save_settings_log(values)

            # 現在の設定も更新
            self.sys = new_params
            self.buf_step = int(self.sys.get("buffer_step_down", 10))
            self.display_px = int(self.sys.get("display_size", 640))
            self.win_size = tuple(self.sys.get("window_size", [1600, 900]))

            return True
        except (ValueError, OSError) as e:
            eg.popup_error(f"設定の保存に失敗しました: {e}")
            return False

    def _save_settings_log(self, values: Dict) -> None:
        """設定画面で変更可能なパラメータをタイムスタンプ付きJSONファイルに保存"""
        logs_setting_dir = self.root / "Anomaly_Detection" / "logs_setting"
        logs_setting_dir.mkdir(exist_ok=True)

        now = _dt.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        log_path = logs_setting_dir / f"settings_{timestamp}.json"

        settings_data = {
            "timestamp": now.isoformat(),
            "system_params": {
                "buffer_step_down": int(values["-BUFFER_STEP_DOWN-"]),
                "display_size": int(values["-DISPLAY_SIZE-"]),
                "window_size": [
                    int(values["-WINDOW_WIDTH-"]),
                    int(values["-WINDOW_HEIGHT-"]),
                ],
                "max_log_files": int(values["-MAX_LOG_FILES-"]),
            },
            "base_threshold": float(values["-BASE_THRESHOLD-"]),
            "product_thresholds": {},
        }

        for prod in self.list_products():
            threshold_key = f"-THRESH_{prod}-"
            if threshold_key in values:
                settings_data["product_thresholds"][prod] = float(values[threshold_key])

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(settings_data, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # トリガーモード関連
    # -------------------------------------------------------------------------
    def _on_trigger_detected(self, event: TriggerEvent) -> None:
        """
        トリガー検知時のコールバック（別スレッドから呼ばれる）

        Args:
            event: トリガーイベント情報
        """
        self.pending_inspection = True
        self.inspection_frame = event.frame

    def _on_preview_frame(self, frame: np.ndarray) -> None:
        """
        プレビューフレーム更新コールバック（別スレッドから呼ばれる）

        Args:
            frame: プレビュー用フレーム
        """
        self.preview_frame = frame

    def _on_state_change(self, state: str, remaining: float) -> None:
        """
        トリガー状態変化コールバック（別スレッドから呼ばれる）

        Args:
            state: トリガー状態（"waiting" or "disabled"）
            remaining: 無効時間残り（秒）
        """
        self.trigger_state = state
        self.trigger_remaining = remaining

    def _start_trigger_mode(self, win: eg.Window) -> bool:
        """
        トリガーモード開始

        Args:
            win: 検査ウィンドウ

        Returns:
            開始成功時True
        """
        if self.camera is None:
            return False

        # トリガー設定を適用
        trigger_config = self.sys.get("trigger", {})
        self.camera.set_trigger_config(trigger_config)

        # トリガーモード開始（プレビューなし、検査結果のみ表示）
        if self.camera.start_trigger_mode(
            on_trigger=self._on_trigger_detected,
            on_frame=None,
            on_state_change=self._on_state_change,
        ):
            self.trigger_mode = True
            self.trigger_state = TRIGGER_STATE_WAITING
            self.trigger_remaining = 0.0
            win["-STATE-"].update("トリガー: 待機中", text_color="green")
            win["-CAPTURE-"].update(disabled=True)
            return True
        return False

    def _stop_trigger_mode(self, win: eg.Window) -> None:
        """
        トリガーモード停止

        Args:
            win: 検査ウィンドウ
        """
        if self.camera is not None:
            self.camera.stop_trigger_mode()

        self.trigger_mode = False
        self.preview_frame = None
        self.pending_inspection = False
        self.inspection_frame = None
        win["-STATE-"].update("手動: 待機中", text_color="gray")
        win["-CAPTURE-"].update(disabled=False)

    def _make_trigger_settings_window(self) -> eg.Window:
        """トリガーモード設定画面を生成"""
        cfg = self.sys.get("trigger", {})
        roi = cfg.get("detection_roi_rel", [0.0, 1.0, 0.0, 0.10])

        layout = [
            [eg.Text("トリガーモード設定", font=("Arial", 18))],
            [eg.HSeparator()],

            [eg.Text("検知ROI設定（相対座標 0.0〜1.0）", font=("Arial", 14))],
            [
                eg.Text("Y開始:"), eg.InputText(str(roi[0]), key="-ROI_Y0-", size=(8, 1)),
                eg.Text("Y終了:"), eg.InputText(str(roi[1]), key="-ROI_Y1-", size=(8, 1)),
            ],
            [
                eg.Text("X開始:"), eg.InputText(str(roi[2]), key="-ROI_X0-", size=(8, 1)),
                eg.Text("X終了:"), eg.InputText(str(roi[3]), key="-ROI_X1-", size=(8, 1)),
            ],

            [eg.HSeparator()],
            [eg.Text("検知設定", font=("Arial", 14))],
            [
                eg.Text("差分しきい値:"),
                eg.InputText(str(cfg.get("diff_threshold", 25)), key="-DIFF_THRESH-", size=(10, 1)),
                eg.Text("(0-255)"),
            ],
            [
                eg.Text("変化割合しきい値:"),
                eg.InputText(str(cfg.get("change_ratio_threshold", 0.05)), key="-CHANGE_RATIO-", size=(10, 1)),
                eg.Text("(0.0-1.0)"),
            ],

            [eg.HSeparator()],
            [eg.Text("タイミング設定", font=("Arial", 14))],
            [
                eg.Text("撮像遅延:"),
                eg.InputText(str(cfg.get("capture_delay", 0.3)), key="-CAPTURE_DELAY-", size=(10, 1)),
                eg.Text("秒"),
            ],
            [
                eg.Text("トリガー間隔:"),
                eg.InputText(str(cfg.get("trigger_interval", 2.0)), key="-TRIGGER_INTERVAL-", size=(10, 1)),
                eg.Text("秒"),
            ],

            [eg.HSeparator()],
            [
                eg.Button("保存", key="-SAVE_TRIGGER-"),
                eg.Button("キャンセル", key="-CANCEL_TRIGGER-"),
            ],
        ]
        return eg.Window("トリガー設定", layout, finalize=True, size=(500, 700))

    def _save_trigger_settings(self, values: Dict) -> bool:
        """
        トリガー設定を保存

        Args:
            values: 設定画面の入力値

        Returns:
            保存成功時True
        """
        try:
            trigger_config = {
                "detection_roi_rel": [
                    float(values["-ROI_Y0-"]),
                    float(values["-ROI_Y1-"]),
                    float(values["-ROI_X0-"]),
                    float(values["-ROI_X1-"]),
                ],
                "diff_threshold": int(values["-DIFF_THRESH-"]),
                "change_ratio_threshold": float(values["-CHANGE_RATIO-"]),
                "capture_delay": float(values["-CAPTURE_DELAY-"]),
                "trigger_interval": float(values["-TRIGGER_INTERVAL-"]),
            }

            # system_params.jsonを更新
            self.sys["trigger"] = trigger_config
            config_path = self.cfg / "system_params.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.sys, f, indent=2, ensure_ascii=False)

            # カメラにも設定を適用
            if self.camera is not None:
                self.camera.set_trigger_config(trigger_config)

            return True
        except (ValueError, OSError) as e:
            eg.popup_error(f"トリガー設定の保存に失敗しました: {e}")
            return False

    def _run_trigger_settings(self) -> None:
        """トリガー設定画面を実行"""
        settings_win = self._make_trigger_settings_window()
        while True:
            ev, vals = settings_win.read()
            if ev in (eg.WINDOW_CLOSED, "-CANCEL_TRIGGER-"):
                settings_win.close()
                break
            if ev == "-SAVE_TRIGGER-":
                if self._save_trigger_settings(vals):
                    eg.popup("トリガー設定を保存しました")
                    settings_win.close()
                    break

    # -------------------------------------------------------------------------
    # UI画面生成
    # -------------------------------------------------------------------------
    def _make_inspection_window(self, prod: str) -> eg.Window:
        blank = np.zeros((self.display_px, self.display_px, 3), np.uint8)
        blank_png = cv2.imencode(".png", blank)[1].tobytes()
        layout = [
            [
                eg.Image(
                    data=blank_png,
                    key="-MASTER-",
                    size=(self.display_px, self.display_px),
                ),
                eg.Image(
                    data=blank_png,
                    key="-LIVE-",
                    size=(self.display_px, self.display_px),
                ),
            ],
            [eg.Text("手動: 待機中", key="-STATE-", font=("Arial", 24), text_color="gray")],
            [
                eg.Text("閾値:", font=("Arial", 24)),
                eg.Input("", key="-THRESHOLD_INPUT-", size=(10, 1), font=("Arial", 20)),
                eg.Button("適用", key="-APPLY_THRESHOLD-", font=("Arial", 12)),
            ],
            [eg.Text("結果: - ", key="-RESULT-", font=("Arial", 36))],
            [eg.HSeparator()],
            [   
                eg.Text("モード選択", font=("Arial", 18)),
                eg.Button("トリガーモード", key="-MODE_TRIGGER-", font=("Arial", 12)),
                eg.Button("手動モード", key="-MODE_MANUAL-", font=("Arial", 12)),
            ],
            [eg.HSeparator()],
            [
                eg.Text("トリガーモード", font=("Arial", 18)),
                eg.Button("トリガー設定", key="-TRIGGER_SETTINGS-", font=("Arial", 12)),
            ],
            [eg.HSeparator()],
            [
                eg.Text("手動モード", font=("Arial", 18)),
                eg.Button("撮像＆判定", key="-CAPTURE-", font=("Arial", 12)),
            ],
            [eg.HSeparator()],
            [eg.Button("終了", key="-STOP-")],
        ]
        win = eg.Window(
            f"検査中: {prod}", layout, finalize=True, size=self.win_size, resizable=True
        )
        maximize_window_safe(win, self.win_size)
        return win

    # -------------------------------------------------------------------------
    # 画像処理
    # -------------------------------------------------------------------------
    def _to_png_bytes(self, img: np.ndarray) -> bytes:
        """
        画像PNGバイト変換メソッド - OpenCV画像をPNGバイト列に変換

        Args:
            img (np.ndarray): 変換対象のRGB画像

        Returns:
            bytes: PNG形式のバイト列
        """
        # RGB形式の画像をBGR形式に変換してからPNGエンコード
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.imencode(".png", img_bgr)[1].tobytes()

    # -------------------------------------------------------------------------
    # 撮影・推論
    # -------------------------------------------------------------------------
    def _capture_single_frame(self, timeout: float = 5.0) -> np.ndarray | None:
        """1枚撮影して返す"""
        with self.camera_lock:
            if self.camera is None:
                return None

            self.camera._alloc_buffers(10)
            self.camera.cam.StartStreaming()

            start_time = time.time()
            frame = None

            while time.time() - start_time < timeout:
                img = self.camera.grab_nonblock()
                if img is not None:
                    frame = img
                    break
                time.sleep(0.001)

            self.camera.cam.StopStreaming()
            return frame

    def _run_inspection(self, frame: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, str]:
        """
        1枚の画像に対して推論を実行

        Args:
            frame: 検査対象の画像（RGB形式）

        Returns:
            Tuple[float, np.ndarray, np.ndarray, str]: (スコア, 画像, ヒートマップ, 判定結果)
        """
        start_time = time.time()
        # inspect_minはリストを受け取るため、1枚の画像をリストに
        score, img, heatmap, res = self.detector.inspect_min([frame])
        inference_time = time.time() - start_time
        print(f"推論時間: {inference_time:.3f}s, スコア: {score:.4f}, 結果: {res}")
        return score, img, heatmap, res

    # -------------------------------------------------------------------------
    # UI更新・ログ管理
    # -------------------------------------------------------------------------
    def _manage_log_storage(self):
        """
        ログストレージ管理メソッド - ログファイルの上限管理と古いファイルの削除
        """
        max_logs = int(self.sys.get("max_log_files", 1000))  # 上限、デフォルト1000件
        log_files = sorted([p for p in self.log.glob("*.txt")], key=lambda p: p.name)

        while len(log_files) >= max_logs:
            oldest_log = log_files.pop(0)
            # .txt と .bmp をペアで削除
            oldest_log.unlink()  # .txt
            bmp_path = oldest_log.with_suffix(".bmp")
            if bmp_path.exists():
                bmp_path.unlink()

    def _update_inspection_ui(
        self, win: eg.Window, score: float, img: np.ndarray, heatmap: np.ndarray, res: str
    ):
        """
        検査UI更新メソッド - 検査結果をメインウィンドウに反映

        Args:
            win (eg.Window): 更新対象のウィンドウ
            score (float): 異常スコア
            img (np.ndarray): 検査画像（RGB形式）
            heatmap (np.ndarray): ヒートマップ付き画像（RGB形式）
            res (str): 判定結果 ("OK" or "NG")
        """
        # ヒートマップ画像を表示用にパディング
        padded_img = pad_to_square(heatmap, self.display_px)
        img_png = self._to_png_bytes(padded_img)

        win["-LIVE-"].update(data=img_png)

        # 結果表示
        if res == "OK":
            result_text = f"結果: OK ({score:.3f})"
            result_color = "darkgreen"
        else:
            result_text = f"結果: NG ({score:.3f})"
            result_color = "red"

        win["-RESULT-"].update(result_text, text_color=result_color)

        # 閾値入力欄を現在値で更新
        threshold = float(self.model_params.get("threshold", 0.0))
        win["-THRESHOLD_INPUT-"].update(f"{threshold:.3f}")

        if res == "NG":
            # NGの場合はブザー音を再生（別スレッドで非同期実行）
            threading.Thread(target=play_buzzer_sound, daemon=True).start()

            # ログ保存
            self._manage_log_storage()
            ts = _dt.datetime.now().strftime("%Y%m%d%H%M%S")
            # ヒートマップ画像をBGR形式に変換してから保存
            heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.log / f"{ts}.bmp"), heatmap_bgr)
            (self.log / f"{ts}.txt").write_text(f"score:{score:.6f}")

    # -------------------------------------------------------------------------
    # メインループ
    # -------------------------------------------------------------------------
    def run(self):
        # ----- ① 品種選択 -----
        sel = self._make_select_window()
        while True:
            ev, vals = sel.read()
            # ----- アプリ終了 -----
            if ev in (eg.WINDOW_CLOSED, "-EXIT-"):
                sel.close()
                return

            # ----- 検査開始 -----
            if ev == "-START-" and vals.get("-LIST-"):
                product = vals["-LIST-"][0]
                sel.close()
                break

            # ----- 設定画面 -----
            if ev == "-SETTINGS-":
                settings_win = self._make_settings_window()
                while True:
                    s_ev, s_vals = settings_win.read()
                    if s_ev in (eg.WINDOW_CLOSED, "-CANCEL_SETTINGS-"):
                        settings_win.close()
                        break
                    if s_ev == "-SAVE_SETTINGS-":
                        if self._save_settings(s_vals):
                            eg.popup("設定を保存しました")
                            settings_win.close()
                            break
                    # リセットボタンの処理
                    if s_ev and s_ev.startswith("-RESET_"):
                        prod_name = s_ev[7:-1]  # "-RESET_" と "-" を除去
                        base_threshold = float(s_vals["-BASE_THRESHOLD-"])
                        settings_win[f"-THRESH_{prod_name}-"].update(str(base_threshold))

        # ----- ② モデル & 設定ロード -----
        self.load_product(product)

        # ----- ③ カメラ初期化 -----
        if not self._initialize_camera():
            eg.popup_error("カメラの初期化に失敗しました。カメラ接続を確認してください。")
            return

        # ----- ④ 検査ウィンドウ -----
        win = self._make_inspection_window(product)
        if self.master_padded is not None:
            win["-MASTER-"].update(data=self._to_png_bytes(self.master_padded))
            self.master_padded = None  # 次回以降スキップ

        # 閾値の初期表示（入力欄に現在値をセット）
        threshold = float(self.model_params.get("threshold", 0.0))
        win["-THRESHOLD_INPUT-"].update(f"{threshold:.3f}")

        # ----- ⑤ GUI イベントループ -----
        while True:
            ev, _ = win.read(timeout=50)  # 50msに短縮（プレビュー更新のため）

            # ----- アプリ終了 -----
            if ev in (eg.WINDOW_CLOSED, "-STOP-"):
                if self.trigger_mode:
                    self._stop_trigger_mode(win)
                break

            # ----- 手動モード切替 -----
            if ev == "-MODE_MANUAL-":
                if self.trigger_mode:
                    self._stop_trigger_mode(win)

            # ----- トリガーモード切替 -----
            if ev == "-MODE_TRIGGER-":
                if not self.trigger_mode:
                    if not self._start_trigger_mode(win):
                        eg.popup_error("トリガーモードの開始に失敗しました")

            # ----- トリガー設定 -----
            if ev == "-TRIGGER_SETTINGS-":
                self._run_trigger_settings()

            # ----- しきい値適用 -----
            if ev == "-APPLY_THRESHOLD-":
                raw = win["-THRESHOLD_INPUT-"].get().strip()
                try:
                    new_threshold = float(raw)
                except ValueError:
                    eg.popup_error(f"無効な値です: {raw}")
                else:
                    # メモリ上の値を更新（次回検査から反映）
                    self.detector._threshold = new_threshold
                    self.model_params["threshold"] = new_threshold
                    # 設定ファイルに永続化
                    base_threshold = float(self.base_model.get("threshold", 0.0))
                    self._save_product_threshold(product, new_threshold, base_threshold)

            # ----- 撮像＆判定ボタン（手動モード時のみ）-----
            if ev == "-CAPTURE-" and not self.is_capturing and not self.trigger_mode:
                self.is_capturing = True
                win["-CAPTURE-"].update(disabled=True)
                win["-STATE-"].update("手動: 撮影中", text_color="blue")
                win.refresh()

                # 撮影
                frame = self._capture_single_frame()

                if frame is not None:
                    win["-STATE-"].update("手動: 検査中", text_color="orange")
                    win.refresh()

                    # 推論
                    score, img, heatmap, res = self._run_inspection(frame)

                    # UI更新
                    self._update_inspection_ui(win, score, img, heatmap, res)
                else:
                    win["-RESULT-"].update("結果: 撮影失敗", text_color="red")

                win["-STATE-"].update("手動: 待機中", text_color="gray")
                win["-CAPTURE-"].update(disabled=False)
                self.is_capturing = False

            # ----- トリガーモード: 状態表示更新 -----
            if self.trigger_mode and not self.pending_inspection:
                if self.trigger_state == TRIGGER_STATE_DISABLED:
                    win["-STATE-"].update(
                        f"トリガー: 無効中 (残り {self.trigger_remaining:.1f}秒)",
                        text_color="red"
                    )
                else:
                    win["-STATE-"].update("トリガー: 待機中", text_color="green")

            # ----- トリガーモード: 検査実行 -----
            if self.trigger_mode and self.pending_inspection:
                self.pending_inspection = False
                frame = self.inspection_frame
                self.inspection_frame = None

                if frame is not None:
                    win["-STATE-"].update("トリガー: 検査中", text_color="orange")
                    win.refresh()

                    # 推論
                    score, img, heatmap, res = self._run_inspection(frame)

                    # UI更新
                    self._update_inspection_ui(win, score, img, heatmap, res)

        # カメラクリーンアップ
        self._cleanup_camera()

        win.close()
        eg.popup("アプリを終了しました。")


def main():
    """エントリポイント"""
    app = AnomalyInspectionApp(Path(__file__).parent.parent)
    app.run()


if __name__ == "__main__":
    main()
