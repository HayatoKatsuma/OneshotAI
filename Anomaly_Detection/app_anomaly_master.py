#!/usr/bin/env python3
# Anomaly_Detection/app_anomaly_master.py
"""
マスター画像撮像 + 特徴量生成 統合アプリ（TkEasyGUI版）

機能:
- Baumerカメラでマスター画像を撮影・プレビュー表示
- master_YYYYMMDD_HHMMSS.bmp として保存
- DINOv2特徴量(master.npy) + 平均Hue(master_hue.txt) を生成
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import TkEasyGUI as eg
from PIL import Image, ImageOps

# プロジェクトルートをパスに追加
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Anomaly_Detection.src.backbones import get_model
from utils.baumer_camera import BaumerCamera


# =============================================================================
# ヘルパー関数
# =============================================================================
def _pad_to_square(img: np.ndarray, size: int) -> np.ndarray:
    """画像を正方形にパディング・リサイズ"""
    pil = Image.fromarray(img)
    pil = ImageOps.pad(pil, (size, size), color=(0, 0, 0))
    return np.asarray(pil)


def _to_png_bytes(img: np.ndarray) -> bytes:
    """RGB画像をPNGバイト列に変換"""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".png", img_bgr)[1].tobytes()


# =============================================================================
# AnomalyMasterApp クラス
# =============================================================================
class AnomalyMasterApp:
    """マスター画像撮像 + 特徴量生成 統合アプリ"""

    DISPLAY_PX = 480

    def __init__(self):
        self.root = _project_root
        self.cfg = self.root / "config"
        self.anomaly_cfg = self.root / "Anomaly_Detection" / "config"
        self.master_dir = self.root / "Anomaly_Detection" / "master"
        self.master_dir.mkdir(parents=True, exist_ok=True)

        # 設定読み込み
        self.camera_params = self._load_json(self.cfg / "base_camera_params.json")
        self.model_params = self._load_json(self.anomaly_cfg / "base_model_params.json")

        # カメラ
        self.camera: BaumerCamera | None = None

        # 状態
        self.is_processing = False
        self.current_product: str | None = None

        eg.set_theme("clam")

    # -------------------------------------------------------------------------
    # JSON読み込み
    # -------------------------------------------------------------------------
    @staticmethod
    def _load_json(path: Path) -> Dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------------------------------------------------
    # カメラ
    # -------------------------------------------------------------------------
    def _initialize_camera(self) -> bool:
        """カメラを初期化・設定適用"""
        print("カメラを初期化中...")
        self.camera = BaumerCamera()
        self.camera.apply_config(self.camera_params)
        print("カメラの初期化が完了しました")
        return True

    def _cleanup_camera(self):
        """カメラを切断"""
        if self.camera is None:
            return
        if self.camera.cam.IsStreaming():
            self.camera.cam.StopStreaming()
        self.camera.cam.Disconnect()
        self.camera = None
        print("カメラを切断しました")

    # -------------------------------------------------------------------------
    # 製品フォルダ
    # -------------------------------------------------------------------------
    def _list_products(self) -> List[str]:
        """master/ 内の製品フォルダ一覧を取得"""
        products = []
        if not self.master_dir.exists():
            return products
        for folder in self.master_dir.iterdir():
            if folder.is_dir():
                products.append(folder.name)
        return sorted(products)

    def _get_bmp_list(self) -> List[str]:
        """現在の製品フォルダ内のBMPファイル名一覧"""
        if not self.current_product:
            return []
        product_dir = self.master_dir / self.current_product
        if not product_dir.exists():
            return []
        return sorted([p.name for p in product_dir.glob("*.bmp")])

    # -------------------------------------------------------------------------
    # 撮像
    # -------------------------------------------------------------------------
    def _capture_and_save(self, win: eg.Window) -> None:
        """1枚撮影してBMP保存・UI表示"""
        if self.camera is None or self.current_product is None:
            return

        win["-STATUS-"].update("撮影中...")
        win.refresh()

        frame = self.camera.grab_nonblock()
        if frame is None:
            # リトライ（最大0.5秒）
            start = time.time()
            while time.time() - start < 0.5:
                frame = self.camera.grab_nonblock()
                if frame is not None:
                    break
                time.sleep(0.01)

        if frame is None:
            eg.popup_error("撮像に失敗しました（タイムアウト）")
            win["-STATUS-"].update("撮像失敗")
            return

        # UI表示
        padded = _pad_to_square(frame, self.DISPLAY_PX)
        win["-PREVIEW-"].update(data=_to_png_bytes(padded))

        # 保存
        product_dir = self.master_dir / self.current_product
        product_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"master_{timestamp}.bmp"
        filepath = product_dir / filename
        image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), image_bgr)

        # BMP一覧を更新
        win["-BMP_LIST-"].update(self._get_bmp_list())
        win["-STATUS-"].update(f"保存: {filename}")

    # -------------------------------------------------------------------------
    # 画像削除
    # -------------------------------------------------------------------------
    def _delete_selected_images(self, win: eg.Window, selected: List[str]) -> None:
        """選択されたBMPファイルを削除"""
        if not selected or not self.current_product:
            return
        product_dir = self.master_dir / self.current_product
        for name in selected:
            path = product_dir / name
            if path.exists():
                path.unlink()
                print(f"削除: {path}")
        win["-BMP_LIST-"].update(self._get_bmp_list())
        win["-STATUS-"].update(f"{len(selected)}件の画像を削除しました")

    # -------------------------------------------------------------------------
    # 特徴量生成（DINOv2 + Hue）
    # -------------------------------------------------------------------------
    @staticmethod
    def _crop_roi_rel(
        img_bgr: np.ndarray, roi_rel: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """相対ROIで矩形切り出し（BGR前提）"""
        h, w = img_bgr.shape[:2]
        y0 = int(roi_rel[0] * h)
        y1 = int(roi_rel[1] * h)
        x0 = int(roi_rel[2] * w)
        x1 = int(roi_rel[3] * w)
        return img_bgr[y0:y1, x0:x1]

    @staticmethod
    def _circular_mean_deg(hue_deg_1d: np.ndarray) -> float:
        """角度データの円環平均（度）"""
        rad = np.deg2rad(hue_deg_1d)
        c = np.cos(rad).mean()
        s = np.sin(rad).mean()
        if c == 0 and s == 0:
            return float("nan")
        return float(math.degrees(math.atan2(s, c)) % 360.0)

    def _compute_master_hue_from_folder(
        self,
        master_folder: Path,
        roi_rel: Tuple[float, float, float, float],
        sat_thresh: int,
    ) -> float:
        """マスターフォルダ内の全BMPから平均Hueを円環平均で計算"""
        bmp_paths = sorted(master_folder.glob("*.bmp"))
        if not bmp_paths:
            return float("nan")

        hue_means = []
        for p in bmp_paths:
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            roi = self._crop_roi_rel(img_bgr, roi_rel)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].astype(np.float32) * 2.0  # 0..360
            sat = hsv[:, :, 1].astype(np.float32)
            mask = sat > float(sat_thresh)
            if np.any(mask):
                hue_means.append(self._circular_mean_deg(hue[mask]))

        if not hue_means:
            return float("nan")
        return self._circular_mean_deg(np.array(hue_means))

    def _generate_features_thread(self, product: str, window: eg.Window) -> None:
        """別スレッドで特徴量生成を実行"""
        import torch

        master_folder = self.master_dir / product
        bmp_files = sorted(master_folder.glob("*.bmp"))
        if not bmp_files:
            window.post_event("-ERROR-", {"-ERROR-": "BMPファイルが見つかりません"})
            return

        total_steps = len(bmp_files) + 2  # BMP枚数 + Hue計算 + 保存
        step = 0

        # --- DINOv2 特徴量生成 ---
        window.post_event("-UPDATE-", {"-UPDATE-": "モデルを初期化中..."})
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(
            self.model_params["model_type"],
            device,
            smaller_edge_size=self.model_params["image_size"],
            feat_layer=self.model_params["feat_layer"],
        )

        all_features = []
        for bmp_path in bmp_files:
            if not self.is_processing:
                window.post_event("-UPDATE-", {"-UPDATE-": "中断しました"})
                return
            window.post_event("-UPDATE-", {"-UPDATE-": f"特徴抽出: {bmp_path.name}"})
            master_image = cv2.cvtColor(
                cv2.imread(str(bmp_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            master_tensor, _ = model.prepare_image(master_image)
            feature = model.extract_features(master_tensor)
            all_features.append(feature)
            step += 1
            progress = int((step / total_steps) * 100)
            window.post_event("-PROGRESS-", {"-PROGRESS-": progress})

        # 特徴量を連結して保存
        concatenated = np.concatenate(all_features, axis=0)
        npy_path = master_folder / "master.npy"
        np.save(npy_path, concatenated)
        print(f"特徴量保存: {npy_path} (patches={concatenated.shape[0]})")

        # --- Hue計算 ---
        window.post_event("-UPDATE-", {"-UPDATE-": "Hue計算中..."})
        roi_rel = tuple(self.model_params.get("roi_rel", [0.0, 1.0, 0.0, 1.0]))
        sat_thresh = int(self.model_params.get("sat_thresh", 30))
        hue_value = self._compute_master_hue_from_folder(master_folder, roi_rel, sat_thresh)
        hue_path = master_folder / "master_hue.txt"
        with open(hue_path, "w", encoding="utf-8") as f:
            f.write(f"{hue_value:.6f}\n")
        print(f"Hue保存: {hue_path} (value={hue_value:.6f})")
        step += 1

        # 完了
        step += 1
        window.post_event("-PROGRESS-", {"-PROGRESS-": 100})
        window.post_event(
            "-COMPLETE-",
            {"-COMPLETE-": f"特徴量生成完了: {len(bmp_files)}枚 → master.npy + master_hue.txt"},
        )
        self.is_processing = False
        window.post_event("-ENABLE-", {"-ENABLE-": True})

    # -------------------------------------------------------------------------
    # プログレスバー
    # -------------------------------------------------------------------------
    @staticmethod
    def _update_progress_bar(window: eg.Window, progress: int) -> None:
        """Canvas上にプログレスバーを描画"""
        canvas = window["-PROGRESSBAR-"]
        canvas.delete("all")
        bar_width, bar_height = 380, 15
        x_off, y_off = 5, 5
        canvas.create_rectangle(
            x_off, y_off, x_off + bar_width, y_off + bar_height,
            outline="black", fill="lightgray", width=2,
        )
        if progress > 0:
            pw = int((progress / 100) * bar_width)
            canvas.create_rectangle(
                x_off + 1, y_off + 1, x_off + pw - 1, y_off + bar_height - 1,
                fill="blue", outline="",
            )
        canvas.create_text(
            x_off + bar_width // 2, y_off + bar_height // 2,
            text=f"{progress}%", font=("Arial", 10), fill="black",
        )

    # -------------------------------------------------------------------------
    # GUI: 製品選択画面
    # -------------------------------------------------------------------------
    def _run_product_select(self) -> str | None:
        """製品選択ダイアログ。戻り値は製品名、キャンセル時はNone。"""
        products = self._list_products()
        layout = [
            [eg.Text("マスター画像管理", font=("Arial", 18))],
            [eg.HSeparator()],
            [eg.Text("既存製品を選択:")],
            [eg.Listbox(products, size=(40, 8), key="-PRODUCTS-")],
            [
                eg.Button("選択", key="-SELECT-"),
                eg.Button("新規作成", key="-NEW-"),
                eg.Button("終了", key="-EXIT-"),
            ],
        ]
        win = eg.Window("製品選択", layout, finalize=True, size=(500, 400))

        result = None
        while True:
            ev, vals = win.read()
            if ev in (eg.WINDOW_CLOSED, "-EXIT-"):
                break
            if ev == "-SELECT-" and vals.get("-PRODUCTS-"):
                result = vals["-PRODUCTS-"][0]
                break
            if ev == "-NEW-":
                name = eg.popup_get_text("新しい製品名を入力:")
                if name and name.strip():
                    name = name.strip()
                    product_dir = self.master_dir / name
                    product_dir.mkdir(parents=True, exist_ok=True)
                    result = name
                    break
        win.close()
        return result

    # -------------------------------------------------------------------------
    # GUI: メイン画面
    # -------------------------------------------------------------------------
    def _make_main_window(self, product: str) -> eg.Window:
        """メイン画面を作成"""
        blank = np.zeros((self.DISPLAY_PX, self.DISPLAY_PX, 3), np.uint8)
        blank_png = _to_png_bytes(blank)
        bmp_list = self._get_bmp_list()

        layout = [
            [eg.Text(f"マスター管理: {product}", font=("Arial", 18))],
            [eg.HSeparator()],
            [
                # 左: プレビュー
                eg.Column([
                    [eg.Text("撮影プレビュー", font=("Arial", 12))],
                    [eg.Image(data=blank_png, key="-PREVIEW-", size=(self.DISPLAY_PX, self.DISPLAY_PX))],
                    [
                        eg.Button("撮影", key="-CAPTURE-", size=(12, 2)),
                    ],
                ]),
                # 右: 画像一覧と特徴量生成
                eg.Column([
                    [eg.Text("保存済みBMP一覧", font=("Arial", 12))],
                    [eg.Listbox(bmp_list, select_mode="multiple", size=(35, 12), key="-BMP_LIST-")],
                    [
                        eg.Button("選択画像を削除", key="-DELETE-", size=(14, 1)),
                    ],
                    [eg.HSeparator()],
                    [eg.Text("特徴量生成", font=("Arial", 12))],
                    [
                        eg.Button("特徴量生成", key="-GENERATE-", size=(14, 2), button_color=("white", "green")),
                        eg.Button("停止", key="-STOP_GEN-", size=(8, 2), button_color=("white", "red"), disabled=True),
                    ],
                    [eg.Canvas(size=(400, 25), key="-PROGRESSBAR-", background_color="white")],
                ]),
            ],
            [eg.HSeparator()],
            [eg.Text("", key="-STATUS-", size=(80, 1), font=("Arial", 10))],
            [eg.Button("終了", key="-BACK-", size=(10, 1))],
        ]
        return eg.Window(f"マスター管理: {product}", layout, finalize=True, size=(1100, 750))

    # -------------------------------------------------------------------------
    # メインループ
    # -------------------------------------------------------------------------
    def run(self):
        """アプリケーション実行"""
        # カメラ初期化
        if not self._initialize_camera():
            eg.popup_error("カメラの初期化に失敗しました")
            return

        while True:
            # 製品選択
            product = self._run_product_select()
            if product is None:
                break

            self.current_product = product
            win = self._make_main_window(product)
            self._update_progress_bar(win, 0)
            win["-STATUS-"].update("準備完了")

            # イベントループ
            while True:
                ev, vals = win.read(timeout=100)

                if ev in (eg.WINDOW_CLOSED, "-BACK-"):
                    if self.is_processing:
                        self.is_processing = False
                    break

                # ----- 撮影 -----
                if ev == "-CAPTURE-":
                    self._capture_and_save(win)

                # ----- 画像削除 -----
                if ev == "-DELETE-":
                    selected = vals.get("-BMP_LIST-", [])
                    if selected:
                        self._delete_selected_images(win, selected)

                # ----- 特徴量生成開始 -----
                if ev == "-GENERATE-":
                    bmp_list = self._get_bmp_list()
                    if not bmp_list:
                        eg.popup_error("BMPファイルがありません。先に撮影してください。")
                        continue
                    self.is_processing = True
                    win["-GENERATE-"].update(disabled=True)
                    win["-STOP_GEN-"].update(disabled=False)
                    self._update_progress_bar(win, 0)
                    threading.Thread(
                        target=self._generate_features_thread,
                        args=(product, win),
                        daemon=True,
                    ).start()

                # ----- 特徴量生成停止 -----
                if ev == "-STOP_GEN-":
                    self.is_processing = False
                    win["-GENERATE-"].update(disabled=False)
                    win["-STOP_GEN-"].update(disabled=True)
                    win["-STATUS-"].update("処理を中断しました")

                # ----- 別スレッドからのイベント -----
                if ev == "-UPDATE-":
                    msg = vals[ev] if isinstance(vals[ev], str) else vals[ev].get("-UPDATE-", "")
                    win["-STATUS-"].update(msg)

                if ev == "-PROGRESS-":
                    prog = vals[ev] if isinstance(vals[ev], int) else vals[ev].get("-PROGRESS-", 0)
                    self._update_progress_bar(win, prog)

                if ev == "-COMPLETE-":
                    win["-GENERATE-"].update(disabled=False)
                    win["-STOP_GEN-"].update(disabled=True)
                    self._update_progress_bar(win, 100)
                    msg = vals[ev] if isinstance(vals[ev], str) else vals[ev].get("-COMPLETE-", "")
                    win["-STATUS-"].update(msg)
                    eg.popup_ok("完了", msg)

                if ev == "-ERROR-":
                    win["-GENERATE-"].update(disabled=False)
                    win["-STOP_GEN-"].update(disabled=True)
                    msg = vals[ev] if isinstance(vals[ev], str) else vals[ev].get("-ERROR-", "")
                    win["-STATUS-"].update(f"エラー: {msg}")
                    eg.popup_error("エラー", msg)

                if ev == "-ENABLE-":
                    win["-GENERATE-"].update(disabled=False)
                    win["-STOP_GEN-"].update(disabled=True)

            win.close()

        # カメラクリーンアップ
        self._cleanup_camera()


def main():
    app = AnomalyMasterApp()
    app.run()


if __name__ == "__main__":
    main()
