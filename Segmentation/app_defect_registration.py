#!/usr/bin/env python3
# Segmentation/app_defect_registration.py
"""
欠陥サンプル登録GUIアプリケーション

機能:
- 画像ファイルの読み込み / カメラ撮影
- BBOXのドラッグ描画
- 欠陥カテゴリの作成・管理
- サンプル登録
"""

from __future__ import annotations
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import TkEasyGUI as eg
import cv2
import numpy as np

# プロジェクトルートをパスに追加
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Segmentation.utils.defect_library import (
    DefectLibrary,
    DefectType,
    create_new_library,
    list_available_libraries,
)
from Segmentation.utils.visualization import pad_to_square


class DefectRegistrationApp:
    """欠陥サンプル登録GUIアプリケーション"""

    def __init__(self, root: Path):
        self.root = root
        self.cfg = root / "config"  # 共有設定（カメラ等）
        self.segmentation_cfg = root / "Segmentation" / "config"  # 欠陥検出専用設定
        self.defect_library_root = root / "Segmentation" / "defect_library"

        # 設定読み込み
        self.defect_params = self._load_json("defect_params.json", use_segmentation_cfg=True)
        self.camera_params = self._load_json("base_camera_params.json")  # カメラ設定（共有）
        self.display_px = int(self.defect_params.get("display_size", 640))

        # 状態
        self.library: Optional[DefectLibrary] = None
        self.current_image: Optional[np.ndarray] = None
        self.current_bbox: Optional[Tuple[float, float, float, float]] = None

        # BBOX描画用
        self.bbox_start: Optional[Tuple[int, int]] = None
        self.bbox_end: Optional[Tuple[int, int]] = None

        # カメラ（遅延初期化）
        self.camera = None
        self.camera_ready = False

        # 画像表示用（参照保持）
        self._tk_image = None

        eg.set_theme("clam")

    def _initialize_camera(self) -> bool:
        """カメラを初期化してストリーミング準備"""
        if self.camera_ready:
            return True

        from utils.baumer_camera import BaumerCamera

        print("カメラを初期化中...")
        self.camera = BaumerCamera()
        self.camera.apply_config(self.camera_params)
        self.camera.cam.StartStreaming()
        self.camera_ready = True
        print("カメラ初期化完了")
        return True

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

    def _load_json(self, filename: str, use_segmentation_cfg: bool = False) -> Dict:
        """
        設定ファイル読み込み（// コメント対応）

        Args:
            filename: ファイル名
            use_segmentation_cfg: Trueの場合、欠陥検出専用設定フォルダから読み込み
        """
        cfg_dir = self.segmentation_cfg if use_segmentation_cfg else self.cfg
        path = cfg_dir / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            lines = [line for line in f if not line.strip().startswith("//")]
            return json.loads("".join(lines))

    def _make_library_select_window(self) -> eg.Window:
        """ライブラリ選択画面"""
        libraries = list_available_libraries(self.defect_library_root)

        layout = [
            [eg.Text("欠陥ライブラリ管理", font=("Arial", 22))],
            [eg.HSeparator()],
            [eg.Text("既存ライブラリを選択:")],
            [eg.Listbox(libraries, size=(40, 6), key="-LIBRARY_LIST-")],
            [
                eg.Button("選択", key="-SELECT_LIB-"),
                eg.Button("新規作成", key="-NEW_LIB-"),
                eg.Button("削除", key="-DELETE_LIB-"),
                eg.Button("終了", key="-BACK-"),
            ],
        ]

        return eg.Window("ライブラリ選択", layout, finalize=True, size=(500, 400))

    def _make_new_library_dialog(self) -> Optional[str]:
        """新規ライブラリ作成ダイアログ"""
        import time
        layout = [
            [eg.Text("新しいライブラリ名を入力:")],
            [eg.InputText("", key="-LIB_NAME-", size=(30, 1))],
            [eg.Button("作成", key="-CREATE-"), eg.Button("キャンセル", key="-CANCEL-")],
        ]

        win = eg.Window("新規ライブラリ作成", layout, finalize=True, size=(350, 150))
        time.sleep(0.1)  # ウィンドウ表示待機

        while True:
            ev, vals = win.read()
            if ev in (eg.WINDOW_CLOSED, "-CANCEL-"):
                win.close()
                return None
            if ev == "-CREATE-":
                name = vals["-LIB_NAME-"].strip()
                if name:
                    win.close()
                    return name
                eg.popup_error("ライブラリ名を入力してください")

    def _make_main_window(self) -> eg.Window:
        """メイン登録画面"""
        # 欠陥タイプリスト
        defect_types = self.library.list_defect_types() if self.library else []

        layout = [
            [eg.Text(f"ライブラリ: {self.library.library_path.name if self.library else '未選択'}", font=("Arial", 18))],
            [eg.HSeparator()],
            # 画像表示（Graphエレメントでマウスイベント対応）
            [
                eg.Column([
                    [eg.Text("サンプル画像（ドラッグでBBOX選択）")],
                    [eg.Canvas(
                        size=(self.display_px, self.display_px),
                        key="-CANVAS-",
                        background_color="black",
                    )],
                    [
                        eg.Button("画像読込", key="-LOAD_IMAGE-"),
                        eg.Button("撮影", key="-CAPTURE-"),
                        eg.Button("BBOX解除", key="-CLEAR_BBOX-"),
                    ],
                ]),
                eg.Column([
                    [eg.Text("欠陥タイプ")],
                    [eg.Listbox(defect_types, size=(20, 6), key="-TYPE_LIST-")],
                    [
                        eg.Button("新規タイプ", key="-NEW_TYPE-"),
                        eg.Button("削除", key="-DEL_TYPE-"),
                    ],
                    [eg.HSeparator()],
                    [eg.Text("BBOX: 画像上でドラッグして選択")],
                    [eg.Text("選択中: なし", key="-BBOX_INFO-", size=(25, 1))],
                    [eg.Button("サンプル登録", key="-REGISTER-", disabled=True)],
                ]),
            ],
            [eg.HSeparator()],
            [eg.Button("終了", key="-BACK-")],
            [eg.Text("※ サンプル登録時に自動保存されます", font=("Arial", 10))],
        ]

        return eg.Window("欠陥サンプル登録", layout, finalize=True, size=(1000, 1200))

    def _make_new_type_dialog(self) -> Optional[Dict]:
        """新規欠陥タイプ作成ダイアログ"""
        import time
        import random
        colors = self.defect_params.get("defect_colors", {})
        display_names = self.defect_params.get("defect_display_names", {})

        preset_types = list(colors.keys())

        # ランダム色生成用のプリセット（視認性の良い色）
        color_palette = [
            (255, 0, 0),      # 赤
            (0, 255, 0),      # 緑
            (0, 0, 255),      # 青
            (255, 255, 0),    # 黄
            (255, 0, 255),    # マゼンタ
            (0, 255, 255),    # シアン
            (255, 128, 0),    # オレンジ
            (128, 0, 255),    # 紫
            (255, 128, 128),  # ライトレッド
            (128, 255, 128),  # ライトグリーン
        ]

        layout = [
            [eg.Text("欠陥タイプ設定", font=("Arial", 14))],
            [eg.HSeparator()],
            [eg.Text("プリセットから選択:")],
            [eg.Combo(preset_types, key="-PRESET-", size=(15, 1))],
            [eg.Text("または、カスタム名（英字）:")],
            [eg.InputText("", key="-CUSTOM_NAME-", size=(20, 1))],
            [eg.Text("表示名（日本語可）:")],
            [eg.InputText("", key="-DISPLAY_NAME-", size=(20, 1))],
            [eg.Text("確信度閾値（0-1）:")],
            [eg.InputText("0.75", key="-THRESHOLD-", size=(10, 1))],
            [eg.Text("サイズ許容範囲（%）:")],
            [eg.InputText("30", key="-SIZE_TOLERANCE-", size=(10, 1))],
            [eg.HSeparator()],
            [eg.Button("作成", key="-CREATE-"), eg.Button("キャンセル", key="-CANCEL-")],
        ]

        win = eg.Window("新規欠陥タイプ", layout, finalize=True, size=(350, 450))
        time.sleep(0.1)  # ウィンドウ表示待機

        while True:
            ev, vals = win.read()
            if ev in (eg.WINDOW_CLOSED, "-CANCEL-"):
                win.close()
                return None

            if ev == "-CREATE-":
                # 名前決定
                preset = vals["-PRESET-"]
                custom = vals["-CUSTOM_NAME-"].strip()
                name = custom if custom else preset

                if not name:
                    eg.popup_error("タイプ名を選択または入力してください")
                    continue

                # 表示名
                display_name = vals["-DISPLAY_NAME-"].strip()
                if not display_name:
                    display_name = display_names.get(name, name)

                # 色（プリセットにあればその色、なければランダム）
                if name in colors:
                    color = tuple(colors[name])
                else:
                    color = random.choice(color_palette)

                # 閾値
                threshold = float(vals["-THRESHOLD-"])

                # サイズ許容範囲（%を0-1に変換）
                size_tolerance = float(vals["-SIZE_TOLERANCE-"]) / 100.0
                size_tolerance = max(0.0, min(1.0, size_tolerance))  # 0-1にクランプ

                win.close()
                return {
                    "name": name,
                    "display_name": display_name,
                    "color": color,
                    "threshold": threshold,
                    "size_tolerance": size_tolerance,
                }

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

    def _capture_image(self) -> Optional[np.ndarray]:
        """BaumerCameraで撮像（事前初期化されたカメラを使用）"""
        import time

        # カメラが初期化されていない場合は初期化
        if not self.camera_ready:
            if not self._initialize_camera():
                eg.popup_error("カメラの初期化に失敗しました")
                return None

        # 画像取得を試行（最大0.5秒待機）
        start_time = time.time()
        while time.time() - start_time < 0.5:
            image = self.camera.grab_nonblock()
            if image is not None:
                return image  # RGB形式
            time.sleep(0.01)  # 10ms待機

        eg.popup_error("撮像に失敗しました（タイムアウト）")
        return None

    def _update_image_display(self, win: eg.Window, temp_bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
        """
        画像表示を更新（Canvas要素を使用）

        Args:
            win: ウィンドウ
            temp_bbox: 一時的なBBOX（ドラッグ中）ピクセル座標 (x1, y1, x2, y2)
        """
        from PIL import Image, ImageTk

        canvas = win["-CANVAS-"].widget

        # キャンバスをクリア
        canvas.delete("all")

        if self.current_image is None:
            return

        # 画像をリサイズしてPIL Imageに変換
        padded = pad_to_square(self.current_image, self.display_px)
        pil_image = Image.fromarray(padded)
        self._tk_image = ImageTk.PhotoImage(pil_image)  # 参照を保持

        # 画像を描画
        canvas.create_image(0, 0, anchor="nw", image=self._tk_image)

        # 確定済みBBOXがあれば描画（緑色）- 元画像座標から表示座標に変換
        if self.current_bbox is not None:
            orig_h, orig_w = self.current_image.shape[:2]
            scale, pad_x, pad_y = self._get_padding_info((orig_h, orig_w), self.display_px)

            cx, cy, bw, bh = self.current_bbox
            # 元画像での座標
            x1_orig = (cx - bw / 2) * orig_w
            y1_orig = (cy - bh / 2) * orig_h
            x2_orig = (cx + bw / 2) * orig_w
            y2_orig = (cy + bh / 2) * orig_h

            # 表示座標に変換
            x1 = int(x1_orig * scale + pad_x)
            y1 = int(y1_orig * scale + pad_y)
            x2 = int(x2_orig * scale + pad_x)
            y2 = int(y2_orig * scale + pad_y)
            canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2)

        # 一時的なBBOX（ドラッグ中）があれば描画（赤色）
        if temp_bbox is not None:
            x1, y1, x2, y2 = temp_bbox
            canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def _get_padding_info(self, original_shape: Tuple[int, int], display_size: int) -> Tuple[float, int, int]:
        """
        パディング情報を計算

        Args:
            original_shape: 元画像の (H, W)
            display_size: 表示サイズ（正方形）

        Returns:
            Tuple[scale, pad_x, pad_y]: スケール、X方向パディング、Y方向パディング
        """
        orig_h, orig_w = original_shape
        scale = min(display_size / orig_w, display_size / orig_h)
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        pad_x = (display_size - scaled_w) // 2
        pad_y = (display_size - scaled_h) // 2
        return scale, pad_x, pad_y

    def _display_to_original_bbox(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        表示座標から元画像の正規化BBOX座標に変換

        Args:
            start: 表示座標の始点 (x, y)
            end: 表示座標の終点 (x, y)

        Returns:
            正規化BBOX座標 (cx, cy, w, h) または範囲外の場合None
        """
        if self.current_image is None:
            return None

        orig_h, orig_w = self.current_image.shape[:2]
        scale, pad_x, pad_y = self._get_padding_info((orig_h, orig_w), self.display_px)

        # 表示座標からパディングを除去してスケーリング座標に変換
        x1_disp = min(start[0], end[0])
        y1_disp = min(start[1], end[1])
        x2_disp = max(start[0], end[0])
        y2_disp = max(start[1], end[1])

        # パディング領域を除外した画像領域
        img_x_start = pad_x
        img_y_start = pad_y
        img_x_end = self.display_px - pad_x
        img_y_end = self.display_px - pad_y

        # BBOXが画像領域内にあるかチェック
        x1_disp = max(x1_disp, img_x_start)
        y1_disp = max(y1_disp, img_y_start)
        x2_disp = min(x2_disp, img_x_end)
        y2_disp = min(y2_disp, img_y_end)

        # 有効なBBOXか確認
        if x2_disp <= x1_disp or y2_disp <= y1_disp:
            return None

        # 表示座標を元画像座標に変換
        x1_orig = (x1_disp - pad_x) / scale
        y1_orig = (y1_disp - pad_y) / scale
        x2_orig = (x2_disp - pad_x) / scale
        y2_orig = (y2_disp - pad_y) / scale

        # 正規化座標に変換
        cx = (x1_orig + x2_orig) / 2 / orig_w
        cy = (y1_orig + y2_orig) / 2 / orig_h
        bw = (x2_orig - x1_orig) / orig_w
        bh = (y2_orig - y1_orig) / orig_h

        # クランプ
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        return (cx, cy, bw, bh)

    def _pixel_to_normalized_bbox(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        img_size: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        """ピクセル座標を正規化BBOX座標に変換（表示座標用、BBOXプレビュー描画に使用）"""
        h, w = img_size
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])

        # 正規化
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        return (cx, cy, bw, bh)

    def _run_library_select(self) -> bool:
        """ライブラリ選択ループ"""
        import time
        import shutil
        win = self._make_library_select_window()
        time.sleep(0.1)  # ウィンドウ表示待機

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
                    self.library.load()
                    win.close()
                    return True

            if ev == "-NEW_LIB-":
                lib_name = self._make_new_library_dialog()
                if lib_name:
                    self.library = create_new_library(self.defect_library_root, lib_name)
                    win.close()
                    return True

            if ev == "-DELETE_LIB-":
                selected = vals.get("-LIBRARY_LIST-", [])
                if selected:
                    lib_name = selected[0]
                    if eg.popup_yes_no(f"ライブラリ '{lib_name}' を削除しますか？\n※この操作は取り消せません") == "Yes":
                        lib_path = self.defect_library_root / lib_name
                        try:
                            shutil.rmtree(lib_path)
                            eg.popup(f"ライブラリ '{lib_name}' を削除しました")
                            # リストを更新
                            libraries = list_available_libraries(self.defect_library_root)
                            win["-LIBRARY_LIST-"].update(values=libraries)
                        except Exception as e:
                            eg.popup_error(f"削除に失敗しました: {e}")
                else:
                    eg.popup_error("削除するライブラリを選択してください")

    def _run_main(self) -> None:
        """メイン登録画面ループ"""
        import time

        # カメラを事前初期化（高速撮像のため）
        self._initialize_camera()

        win = self._make_main_window()
        time.sleep(0.1)  # ウィンドウ表示待機

        # BBOX描画用の状態（nonlocalでアクセスするため辞書で管理）
        drag_state = {"dragging": False, "start": None, "current": None}

        def on_mouse_press(event):
            """マウスボタン押下"""
            if self.current_image is not None:
                drag_state["dragging"] = True
                drag_state["start"] = (event.x, event.y)
                drag_state["current"] = (event.x, event.y)

        def on_mouse_drag(event):
            """マウスドラッグ"""
            if drag_state["dragging"] and self.current_image is not None:
                drag_state["current"] = (event.x, event.y)
                x1, y1 = drag_state["start"]
                x2, y2 = event.x, event.y
                self._update_image_display(win, (x1, y1, x2, y2))

        def on_mouse_release(event):
            """マウスボタン解放"""
            if drag_state["dragging"] and drag_state["start"] is not None:
                x1, y1 = drag_state["start"]
                x2, y2 = event.x, event.y

                # 最小サイズ（10px x 10px）以上の場合のみ確定
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    # 表示座標を元画像の正規化座標に変換
                    original_bbox = self._display_to_original_bbox(
                        (x1, y1), (x2, y2)
                    )

                    if original_bbox is not None:
                        self.current_bbox = original_bbox
                        # 表示更新
                        self._update_image_display(win)

                        # 元画像でのBBOXサイズを表示
                        orig_h, orig_w = self.current_image.shape[:2]
                        _, _, bw, bh = original_bbox
                        bbox_w_px = int(bw * orig_w)
                        bbox_h_px = int(bh * orig_h)
                        win["-BBOX_INFO-"].update(f"選択中: {bbox_w_px}x{bbox_h_px}px (元画像)")
                        win["-REGISTER-"].update(disabled=False)
                    else:
                        # パディング領域を選択した場合
                        eg.popup_error("画像領域内でBBOXを選択してください")
                        self._update_image_display(win)
                else:
                    # 小さすぎる場合は表示を元に戻す
                    self._update_image_display(win)

            drag_state["dragging"] = False
            drag_state["start"] = None
            drag_state["current"] = None

        # Canvasにマウスイベントをバインド
        canvas = win["-CANVAS-"].widget
        canvas.bind("<ButtonPress-1>", on_mouse_press)
        canvas.bind("<B1-Motion>", on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", on_mouse_release)

        while True:
            ev, vals = win.read(timeout=100)

            if ev in (eg.WINDOW_CLOSED, "-BACK-"):
                break

            # BBOX解除
            if ev == "-CLEAR_BBOX-":
                self.current_bbox = None
                self._update_image_display(win)
                win["-BBOX_INFO-"].update("選択中: なし")
                win["-REGISTER-"].update(disabled=True)

            # 画像読み込み
            if ev == "-LOAD_IMAGE-":
                image = self._load_image_from_file()
                if image is not None:
                    self.current_image = image
                    self.current_bbox = None
                    self._update_image_display(win)
                    win["-BBOX_INFO-"].update("選択中: なし")
                    win["-REGISTER-"].update(disabled=True)

            # 撮影（BaumerCameraを使用）
            if ev == "-CAPTURE-":
                image = self._capture_image()
                if image is not None:
                    self.current_image = image
                    self.current_bbox = None
                    self._update_image_display(win)
                    win["-BBOX_INFO-"].update("選択中: なし")
                    win["-REGISTER-"].update(disabled=True)

            # 新規タイプ作成
            if ev == "-NEW_TYPE-":
                type_info = self._make_new_type_dialog()
                if type_info:
                    self.library.add_defect_type(
                        name=type_info["name"],
                        display_name=type_info["display_name"],
                        color=type_info["color"],
                        confidence_threshold=type_info["threshold"],
                        size_tolerance=type_info["size_tolerance"],
                    )
                    # 自動保存
                    self.library.save()
                    # リスト更新
                    win["-TYPE_LIST-"].update(values=self.library.list_defect_types())

            # タイプ削除
            if ev == "-DEL_TYPE-":
                selected = vals.get("-TYPE_LIST-", [])
                if selected:
                    if eg.popup_yes_no(f"'{selected[0]}' を削除しますか？") == "Yes":
                        self.library.remove_defect_type(selected[0])
                        # 自動保存
                        self.library.save()
                        win["-TYPE_LIST-"].update(values=self.library.list_defect_types())

            # サンプル登録
            if ev == "-REGISTER-":
                selected_type = vals.get("-TYPE_LIST-", [])
                if not selected_type:
                    eg.popup_error("欠陥タイプを選択してください")
                    continue

                if self.current_image is None or self.current_bbox is None:
                    eg.popup_error("画像とBBOXを設定してください")
                    continue

                # 元画像を保存（検出時に横方向のみでクロップするため）
                print(f"[DEBUG] 元画像サイズ: {self.current_image.shape}")
                print(f"[DEBUG] BBOX: {self.current_bbox}")

                # サンプル登録（元画像を保存、クロップは検出時に行う）
                sample_id = self.library.add_sample(
                    defect_type_name=selected_type[0],
                    image=self.current_image,
                    bbox=self.current_bbox,
                    # cropped_image を渡さないことで元画像が保存される
                )

                if sample_id:
                    # 自動保存
                    self.library.save()
                    eg.popup(f"サンプルを登録・保存しました (ID: {sample_id})")
                    self.current_bbox = None
                    self._update_image_display(win)
                    win["-BBOX_INFO-"].update("選択中: なし")
                    win["-REGISTER-"].update(disabled=True)

        win.close()

        # カメラ解放
        self._release_camera()

    def run(self) -> None:
        """アプリケーション実行"""
        # ライブラリ選択
        if not self._run_library_select():
            return

        # メイン画面
        self._run_main()


def main():
    """エントリポイント"""
    app = DefectRegistrationApp(Path(__file__).parent.parent)
    app.run()


if __name__ == "__main__":
    main()
