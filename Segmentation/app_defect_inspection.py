#!/usr/bin/env python3
# Segmentation/app_defect_inspection.py
"""
欠陥検出・計数GUIアプリケーション

機能:
- ライブラリ選択
- 撮像＆検出
- 結果表示・CSV出力
"""

from __future__ import annotations
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import TkEasyGUI as eg
import cv2
import numpy as np

# プロジェクトルートをパスに追加
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Segmentation.utils.defect_library import (
    DefectLibrary,
    list_available_libraries,
)
from Segmentation.utils.defect_detector import DefectDetector, DetectionResult
from Segmentation.utils.visualization import pad_to_square, to_png_bytes
from utils.baumer_camera import (
    TriggerEvent,
    TRIGGER_STATE_WAITING,
    TRIGGER_STATE_DISABLED,
)


class DefectInspectionApp:
    """欠陥検出・計数GUIアプリケーション"""

    def __init__(self, root: Path):
        self.root = root
        self.cfg = root / "config"
        self.segmentation_cfg = root / "Segmentation" / "config"
        self.defect_library_root = root / "Segmentation" / "defect_library"

        # 設定読み込み
        self.defect_params = self._load_json("defect_params.json", use_segmentation_cfg=True)
        self.system_params = self._load_json("system_params.json")
        self.camera_params = self._load_json("base_camera_params.json")
        self.display_px = int(self.defect_params.get("display_size", 640))

        # CSV出力ディレクトリ
        csv_dir = self.defect_params.get("csv_output_dir", "Segmentation/logs")
        self.csv_output_dir = root / csv_dir
        self.csv_output_dir.mkdir(parents=True, exist_ok=True)

        # 状態
        self.library: Optional[DefectLibrary] = None
        self.detector: Optional[DefectDetector] = None
        self.camera = None
        self.camera_ready = False
        self.last_result: Optional[DetectionResult] = None

        # CSVファイル管理（アプリ起動ごとに1ファイル）
        self.csv_file_path: Optional[Path] = None
        self.csv_initialized = False

        # トリガーモード関連
        self.trigger_mode: bool = False
        self.preview_frame: Optional[np.ndarray] = None
        self.pending_inspection: bool = False
        self.inspection_frame: Optional[np.ndarray] = None
        self.trigger_state: str = TRIGGER_STATE_WAITING  # トリガー状態
        self.trigger_remaining: float = 0.0              # 無効時間残り（秒）

        eg.set_theme("clam")

    def _load_json(self, filename: str, use_segmentation_cfg: bool = False) -> Dict:
        """設定ファイル読み込み"""
        cfg_dir = self.segmentation_cfg if use_segmentation_cfg else self.cfg
        path = cfg_dir / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _initialize_camera(self) -> bool:
        """カメラを初期化"""
        if self.camera_ready:
            return True

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "baumer_camera", _project_root / "utils" / "baumer_camera.py"
            )
            baumer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(baumer_module)
            BaumerCamera = baumer_module.BaumerCamera

            print("カメラを初期化中...")
            self.camera = BaumerCamera()
            self.camera.apply_config(self.camera_params)
            self.camera.cam.StartStreaming()
            self.camera_ready = True
            print("カメラ初期化完了")
            return True

        except Exception as e:
            print(f"カメラ初期化エラー: {e}")
            self.camera = None
            self.camera_ready = False
            return False

    def _release_camera(self) -> None:
        """カメラを解放"""
        if self.camera is not None:
            try:
                if self.camera.cam.IsStreaming():
                    self.camera.cam.StopStreaming()
                self.camera.cam.Disconnect()
                print("カメラを解放しました")
            except Exception as e:
                print(f"カメラ解放エラー: {e}")
            finally:
                self.camera = None
                self.camera_ready = False

    def _initialize_detector(self, preload: bool = True) -> bool:
        """モデルを初期化"""
        if self.detector is not None and self.detector.is_ready:
            return True

        if self.library is None:
            print("ライブラリが選択されていません")
            return False

        try:
            print("モデルを初期化中...")
            debug_dir = self.csv_output_dir / "debug_images"
            self.detector = DefectDetector(
                self.library,
                device="cuda",
                debug_save_dir=str(debug_dir),
            )

            if preload:
                print("モデルを事前ロード中...")
                if not self.detector.preload_model():
                    print("警告: モデルの事前ロードに失敗しました")
                    return False

            print("モデルの初期化完了")
            return True

        except Exception as e:
            print(f"モデル初期化エラー: {e}")
            self.detector = None
            return False

    def _make_library_select_window(self) -> eg.Window:
        """ライブラリ選択画面"""
        libraries = list_available_libraries(self.defect_library_root)

        layout = [
            [eg.Text("欠陥検出ライブラリ選択", font=("Arial", 22))],
            [eg.HSeparator()],
            [eg.Text("利用可能なライブラリ:")],
            [eg.Listbox(libraries, size=(40, 8), key="-LIBRARY_LIST-")],
            [
                eg.Button("検査開始", key="-SELECT_LIB-"),
            ],
            [eg.HSeparator()],
            [
                eg.Button("終了", key="-BACK-"),
            ],
        ]

        return eg.Window("ライブラリ選択", layout, finalize=True, size=(500, 500))


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
        trigger_config = self.system_params.get("trigger", {})
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
            win["-MODE_STATUS-"].update("トリガー待機中", text_color="green")
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
        win["-MODE_STATUS-"].update("モード: 手動", text_color="gray")
        win["-CAPTURE-"].update(disabled=False)

    def _make_trigger_settings_window(self) -> eg.Window:
        """トリガーモード設定画面を生成"""
        cfg = self.system_params.get("trigger", {})
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
        return eg.Window("トリガー設定", layout, finalize=True, size=(500, 450))

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
            self.system_params["trigger"] = trigger_config
            config_path = self.cfg / "system_params.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.system_params, f, indent=2, ensure_ascii=False)

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

    def _make_main_window(self) -> eg.Window:
        """メイン検査画面"""
        blank = np.zeros((self.display_px, self.display_px, 3), np.uint8)
        blank_png = to_png_bytes(blank)

        type_summary_lines = self._get_defect_type_summary_lines()

        layout = [
            [eg.Text(f"欠陥検出・計数 - {self.library.library_path.name}", font=("Arial", 18))],
            [eg.HSeparator()],
            [
                eg.Column([
                    [eg.Text("検査画像")],
                    [eg.Image(data=blank_png, key="-IMAGE-", size=(self.display_px, self.display_px))],
                ]),
                eg.Column([
                    [eg.Text("検出結果", font=("Arial", 14))],
                    [eg.HSeparator()],
                    [eg.Text("判定:", font=("Arial", 16)), eg.Text("---", key="-JUDGMENT-", font=("Arial", 16, "bold"))],
                    [eg.Text("欠陥総数:"), eg.Text("0", key="-TOTAL_COUNT-", font=("Arial", 14))],
                    [eg.HSeparator()],
                    [eg.Text("検出内訳:", font=("Arial", 12))],
                    [eg.Listbox(["検出なし"], key="-CATEGORY_COUNTS-", size=(30, 5))],
                    [eg.HSeparator()],
                    [eg.Text("登録欠陥クラス:", font=("Arial", 12))],
                    [eg.Listbox(type_summary_lines, key="-TYPE_SUMMARY-", size=(30, 5))],
                ]),
            ],
            [eg.HSeparator()],
            [
                eg.Text("モード: 手動", key="-MODE_STATUS-", font=("Arial", 14), text_color="gray"),
                eg.Button("手動モード", key="-MODE_MANUAL-", size=(10, 1)),
                eg.Button("トリガーモード", key="-MODE_TRIGGER-", size=(12, 1)),
                eg.Button("トリガー設定", key="-TRIGGER_SETTINGS-", size=(10, 1)),
            ],
            [eg.HSeparator()],
            [
                eg.Column([
                    [
                        eg.Button("撮像＆検出", key="-CAPTURE-", size=(12, 2)),
                        eg.Button("画像読込", key="-LOAD_IMAGE-", size=(10, 2)),
                        eg.Button("終了", key="-BACK-", size=(10, 2)),
                    ],
                ]),
            ],
        ]

        return eg.Window("欠陥検出・計数", layout, finalize=True, size=(1100, 950))

    def _get_defect_type_summary(self) -> str:
        """欠陥タイプのサマリーテキストを生成（しきい値含む）"""
        if not self.library:
            return "ライブラリ未選択"

        lines = []
        for defect_type in self.library.defect_types.values():
            threshold = defect_type.confidence_threshold
            lines.append(f"{defect_type.display_name}")
            lines.append(f"    確信度閾値: {threshold:.2f}")

        return "\n".join(lines) if lines else "欠陥クラス未登録"

    def _get_defect_type_summary_lines(self) -> list:
        """欠陥タイプのサマリーをリスト形式で取得（Listbox用）"""
        if not self.library:
            return ["ライブラリ未選択"]

        lines = []
        for defect_type in self.library.defect_types.values():
            threshold = defect_type.confidence_threshold
            lines.append(f" {defect_type.display_name} (閾値: {threshold:.2f})")

        return lines if lines else ["欠陥クラス未登録"]

    def _update_image_display(self, win: eg.Window, image: np.ndarray) -> None:
        """画像表示を更新"""
        padded = pad_to_square(image, self.display_px)
        png_bytes = to_png_bytes(padded)
        win["-IMAGE-"].update(data=png_bytes)

    def _update_result_display(self, win: eg.Window, result: DetectionResult) -> None:
        """検出結果表示を更新"""
        if result.has_defects:
            win["-JUDGMENT-"].update("NG", text_color="red")
        else:
            win["-JUDGMENT-"].update("OK", text_color="green")

        win["-TOTAL_COUNT-"].update(str(result.total_defects))

        # 欠陥クラスごとの内訳を表示
        if result.defect_counts:
            lines = []
            for name, count in result.defect_counts.items():
                lines.append(f"{name}: {count}個")
            win["-CATEGORY_COUNTS-"].update(lines)
        else:
            win["-CATEGORY_COUNTS-"].update(["検出なし"])

        self._update_image_display(win, result.overlay_image)


    def _play_alert_sound(self) -> None:
        """アラート音を再生"""
        sound_file = self.root / "Buzzer.wav"
        if sound_file.exists():
            try:
                subprocess.Popen(
                    ["paplay", str(sound_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    def _capture_and_detect(self, win: eg.Window) -> Optional[DetectionResult]:
        """撮像して検出を実行"""
        import time

        if not self.camera_ready:
            if not self._initialize_camera():
                eg.popup_error("カメラの初期化に失敗しました")
                return None

        image = None
        start_time = time.time()
        while time.time() - start_time < 0.5:
            image = self.camera.grab_nonblock()
            if image is not None:
                break
            time.sleep(0.01)

        if image is None:
            eg.popup_error("撮像に失敗しました（タイムアウト）")
            return None

        return self._detect_image(win, image)

    def _detect_image(self, win: eg.Window, image: np.ndarray) -> Optional[DetectionResult]:
        """画像から欠陥を検出"""
        if self.detector is None or not self.detector.is_ready:
            eg.popup_error("モデルが初期化されていません")
            return None

        result = self.detector.detect(image)
        self.last_result = result

        if result.has_defects:
            self._play_alert_sound()

        self._update_result_display(win, result)

        # 結果画像を保存（NG時のみ）し、CSVに追記
        image_filename = None
        if result.has_defects:
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"result_{timestamp}.png"
            image_path = self.csv_output_dir / "result_images" / image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), cv2.cvtColor(result.overlay_image, cv2.COLOR_RGB2BGR))

        # CSVに検出結果を追記
        self._append_result_to_csv(result, image_filename)

        return result

    def _load_image_from_file(self) -> Optional[np.ndarray]:
        """ファイルから画像を読み込み"""
        file_path = eg.popup_get_file(
            "画像ファイルを選択",
            file_types=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"),),
        )
        if not file_path:
            return None

        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            eg.popup_error(f"画像の読み込みに失敗しました: {file_path}")
            return None

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _initialize_csv_file(self) -> None:
        """アプリ起動時にCSVファイルを初期化"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file_path = self.csv_output_dir / f"defect_inspection_{timestamp}.csv"

        # 欠陥クラス名のリストを取得
        defect_class_names = []
        if self.library:
            defect_class_names = [dt.display_name for dt in self.library.defect_types.values()]

        # ヘッダー行を書き込み
        # 基本列: 検査日時, 判定結果, 総欠陥数, [各欠陥クラス], 結果画像ファイル
        fieldnames = ["検査日時", "判定結果", "総欠陥数"]
        fieldnames.extend(defect_class_names)
        fieldnames.append("結果画像ファイル")

        try:
            with open(self.csv_file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            self.csv_initialized = True
            print(f"CSVファイルを作成しました: {self.csv_file_path}")

        except Exception as e:
            print(f"CSVファイル初期化エラー: {e}")
            self.csv_initialized = False

    def _append_result_to_csv(self, result: DetectionResult, image_path: Optional[str] = None) -> None:
        """検出結果をCSVファイルに追記"""
        if not self.csv_initialized or self.csv_file_path is None:
            print("CSVファイルが初期化されていません")
            return

        # 欠陥クラス名のリストを取得
        defect_class_names = []
        if self.library:
            defect_class_names = [dt.display_name for dt in self.library.defect_types.values()]

        # 行データを作成
        row = {
            "検査日時": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "判定結果": "NG" if result.has_defects else "OK",
            "総欠陥数": result.total_defects,
        }

        # 各欠陥クラスの検出数を追加
        for class_name in defect_class_names:
            row[class_name] = result.defect_counts.get(class_name, 0)

        # 結果画像ファイル名を追加
        row["結果画像ファイル"] = image_path if image_path else ""

        # CSVに追記
        fieldnames = ["検査日時", "判定結果", "総欠陥数"]
        fieldnames.extend(defect_class_names)
        fieldnames.append("結果画像ファイル")

        with open(self.csv_file_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def _run_library_select(self) -> bool:
        """ライブラリ選択ループ"""
        import time
        win = self._make_library_select_window()
        time.sleep(0.1)

        while True:
            ev, vals = win.read()

            if ev in (eg.WINDOW_CLOSED, "-BACK-"):
                win.close()
                return False

            if ev == "-SELECT_LIB-":
                selected = vals.get("-LIBRARY_LIST-", [])
                if selected:
                    lib_name = selected[0]
                    lib_path = self.defect_library_root / lib_name
                    self.library = DefectLibrary(lib_path)
                    if not self.library.load():
                        eg.popup_error("ライブラリの読み込みに失敗しました")
                        continue

                    total_samples = self.library.get_total_sample_count()
                    if total_samples == 0:
                        if eg.popup_yes_no(
                            "このライブラリにはサンプルが登録されていません。\n"
                            "サンプル登録画面に移動しますか？"
                        ) == "Yes":
                            win.close()
                            self._open_registration_app()
                            return False
                        continue

                    win.close()
                    return True
                else:
                    eg.popup_error("ライブラリを選択してください")


    def _open_registration_app(self) -> None:
        """サンプル登録アプリを開く"""
        from Segmentation.app_defect_registration import DefectRegistrationApp
        app = DefectRegistrationApp(self.root)
        app.run()

    def _run_main(self) -> None:
        """メイン検査画面ループ"""
        loading_layout = [
            [eg.Text("初期化中...", font=("Arial", 16), key="-LOADING_TEXT-")],
            [eg.Text("カメラとモデルを準備しています", font=("Arial", 10))],
        ]
        loading_win = eg.Window("準備中", loading_layout, finalize=True, size=(350, 100), no_titlebar=True)

        print("=== 初期化開始 ===")
        loading_win["-LOADING_TEXT-"].update("カメラを初期化中...")
        loading_win.refresh()
        self._initialize_camera()

        loading_win["-LOADING_TEXT-"].update("モデルをロード中...")
        loading_win.refresh()
        self._initialize_detector(preload=True)

        # CSVファイルを初期化
        self._initialize_csv_file()

        print("=== 初期化完了 ===")
        loading_win.close()

        import time
        win = self._make_main_window()
        time.sleep(0.1)

        while True:
            ev, vals = win.read(timeout=50)  # 50msに短縮（プレビュー更新のため）

            # ----- アプリ終了 -----
            if ev in (eg.WINDOW_CLOSED, "-BACK-"):
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

            # ----- 撮像＆検出ボタン（手動モード時のみ）-----
            if ev == "-CAPTURE-" and not self.trigger_mode:
                self._capture_and_detect(win)

            # ----- 画像読込（手動モード時のみ）-----
            if ev == "-LOAD_IMAGE-" and not self.trigger_mode:
                image = self._load_image_from_file()
                if image is not None:
                    self._detect_image(win, image)

            # ----- トリガーモード: 状態表示更新 -----
            if self.trigger_mode and not self.pending_inspection:
                if self.trigger_state == TRIGGER_STATE_DISABLED:
                    win["-MODE_STATUS-"].update(
                        f"トリガー無効中 (残り {self.trigger_remaining:.1f}秒)",
                        text_color="red"
                    )
                else:
                    win["-MODE_STATUS-"].update("トリガー待機中", text_color="green")

            # ----- トリガーモード: 検査実行 -----
            if self.trigger_mode and self.pending_inspection:
                self.pending_inspection = False
                frame = self.inspection_frame
                self.inspection_frame = None

                if frame is not None:
                    self._detect_image(win, frame)

        win.close()

        self._release_camera()
        if self.detector:
            self.detector.unload_model()

    def run(self) -> None:
        """アプリケーション実行"""
        if not self._run_library_select():
            return
        self._run_main()


def main():
    """エントリポイント"""
    app = DefectInspectionApp(Path(__file__).parent.parent)
    app.run()


if __name__ == "__main__":
    main()
