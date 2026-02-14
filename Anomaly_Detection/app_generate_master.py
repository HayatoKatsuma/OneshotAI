#!/usr/bin/env python3
# Anomaly_Detection/app_generate_master.py
"""
マスター特徴量ファイル生成GUI

TkEasyGUIを使用してマスターファイルの特徴量生成を行うGUIアプリ。
以下の機能を提供：
1. config/base_model_params.jsonからモデルパラメータを読み込み
2. masterフォルダの製品一覧をチェックボックスで表示
3. 選択された製品のanomalyDINO特徴量(master.npy)を生成
4. 選択された製品のROI領域平均Hue(master_hue.txt)を計算
"""

import json
import sys
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import TkEasyGUI as sg
import cv2
import numpy as np
import torch

# プロジェクトルートをパスに追加
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Anomaly_Detection.src.backbones import get_model


class MasterGeneratorApp:
    def __init__(self):
        self.root = _project_root
        self.anomaly_dir = self.root / "Anomaly_Detection"
        self.master_dir = self.anomaly_dir / "master"
        self.config_path = self.anomaly_dir / "config" / "base_model_params.json"

        self.model_params = None
        self.is_processing = False
        self.available_products = []
        self.selected_products = []

    def load_model_params(self) -> bool:
        """モデルパラメータをJSONファイルから読み込み"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.model_params = json.load(f)
            return True
        except Exception as e:
            sg.popup_error(f"設定ファイル読み込みエラー: {e}")
            return False

    def get_available_products(self) -> List[str]:
        """masterフォルダから利用可能な製品一覧を取得（1つ以上の.bmpがあるフォルダ）"""
        products = []

        if not self.master_dir.exists():
            return products

        for folder in self.master_dir.iterdir():
            if folder.is_dir():
                # フォルダ内に1つ以上の.bmpファイルがあれば対象
                bmp_files = list(folder.glob("*.bmp"))
                if bmp_files:
                    products.append(folder.name)

        return sorted(products)

    def create_anomaly_dino_features(self, product_name: str, model_params: Dict) -> bool:
        """anomalyDINOの特徴量ファイル(master.npy)を生成（複数画像対応）"""
        try:
            master_folder = self.master_dir / product_name
            npy_path = master_folder / "master.npy"

            # フォルダ内の全.bmpファイルを取得
            bmp_files = sorted(master_folder.glob("*.bmp"))
            if not bmp_files:
                print(f"No BMP files found in {master_folder}")
                return False

            print(f"Found {len(bmp_files)} BMP files for {product_name}")

            # 既存ファイルがあっても上書きする
            if npy_path.exists():
                print(f"Overwriting existing feature file: {npy_path}")

            # デバイス設定
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # モデル初期化
            model = get_model(
                model_params["model_type"],
                device,
                smaller_edge_size=model_params["image_size"],
                feat_layer=model_params["feat_layer"],
            )

            # 全画像から特徴量を抽出して連結
            all_features = []
            for bmp_path in bmp_files:
                print(f"  Processing: {bmp_path.name}")
                master_image = cv2.cvtColor(
                    cv2.imread(str(bmp_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
                master_tensor, _ = model.prepare_image(master_image)
                feature = model.extract_features(master_tensor)
                all_features.append(feature)

            # 全特徴量を連結
            concatenated_features = np.concatenate(all_features, axis=0)
            print(f"  Total patches: {concatenated_features.shape[0]} (from {len(bmp_files)} images)")

            # 保存
            np.save(npy_path, concatenated_features)
            print(f"Successfully created: {npy_path}")
            return True

        except Exception as e:
            print(f"Error creating anomalyDINO features for {product_name}: {e}")
            return False

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
        """角度データの円環平均（度）。"""
        import math

        rad = np.deg2rad(hue_deg_1d)
        c = np.cos(rad).mean()
        s = np.sin(rad).mean()
        if c == 0 and s == 0:
            return float("nan")
        return float((math.degrees(math.atan2(s, c)) % 360.0))

    def _compute_master_hue_from_folder(
        self,
        master_folder_path: Path,
        roi_rel: Tuple[float, float, float, float],
        sat_thresh: int,
    ) -> float:
        """マスター側の平均Hue（度）を、マスター内の全BMPから円環平均で求める"""
        bmp_paths = sorted(master_folder_path.glob("*.bmp"))
        if not bmp_paths:
            print(
                f"[WARN] No BMP found in master folder for Hue computation: {master_folder_path}"
            )
            return float("nan")

        # 各画像のHue平均値を収集
        hue_means = []
        for p in bmp_paths:
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            roi = self._crop_roi_rel(img_bgr, roi_rel)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].astype(np.float32) * 2.0  # 0..360相当
            sat = hsv[:, :, 1].astype(np.float32)
            mask = sat > float(sat_thresh)
            if np.any(mask):
                hue_means.append(self._circular_mean_deg(hue[mask]))

        if not hue_means:
            print(
                f"[WARN] Failed to compute master hue (no valid pixels): {master_folder_path}"
            )
            return float("nan")

        # 複数画像のHue平均値を円環平均で統合
        return self._circular_mean_deg(np.array(hue_means))

    def create_master_hue(self, product_name: str, model_params: Dict) -> bool:
        """master_hue.txtファイルを生成"""
        try:
            master_folder_path = self.master_dir / product_name
            hue_txt_path = master_folder_path / "master_hue.txt"

            # 既存ファイルがあっても上書きする
            if hue_txt_path.exists():
                print(f"Overwriting existing hue file: {hue_txt_path}")

            # ROI領域のHueを計算（analyze_score.pyと同じ方法）
            roi_rel = tuple(model_params["roi_rel"])
            sat_thresh = model_params["sat_thresh"]

            hue_value = self._compute_master_hue_from_folder(
                master_folder_path, roi_rel, sat_thresh
            )

            # ファイルに保存
            with open(hue_txt_path, "w", encoding="utf-8") as f:
                f.write(f"{hue_value:.6f}\n")

            print(f"Successfully created: {hue_txt_path} (value={hue_value:.6f} deg)")
            return True

        except Exception as e:
            print(f"Error creating master_hue for {product_name}: {e}")
            return False

    def process_selected_products(self, selected_products: List[str], window: sg.Window):
        """選択された製品の特徴量生成を別スレッドで実行"""
        try:
            total_products = len(selected_products)

            for i, product in enumerate(selected_products):
                if not self.is_processing:
                    break

                window.post_event("-UPDATE-", {"-UPDATE-": f"Processing: {product}"})

                # anomalyDINO特徴量生成
                success_dino = self.create_anomaly_dino_features(product, self.model_params)

                # ROI Hue計算
                success_hue = self.create_master_hue(product, self.model_params)

                # 進捗更新
                progress = int(((i + 1) / total_products) * 100)
                window.post_event("-PROGRESS-", {"-PROGRESS-": progress})

                if success_dino and success_hue:
                    window.post_event("-UPDATE-", {"-UPDATE-": f"✓ Completed: {product}"})
                else:
                    window.post_event("-UPDATE-", {"-UPDATE-": f"✗ Failed: {product}"})

            window.post_event(
                "-COMPLETE-",
                {"-COMPLETE-": f"Processing completed for {total_products} products"},
            )

        except Exception as e:
            window.post_event("-ERROR-", {"-ERROR-": f"Processing error: {e}"})
        finally:
            self.is_processing = False
            window.post_event("-ENABLE-", {"-ENABLE-": True})

    def update_progress_bar(self, window: sg.Window, progress: int):
        """Canvas を使用してプログレスバーを更新"""
        canvas = window["-PROGRESSBAR-"]
        canvas.delete("all")

        # プログレスバーの設定
        bar_width = 480
        bar_height = 15
        x_offset = 10
        y_offset = 5

        # 背景の枠を描画
        canvas.create_rectangle(
            x_offset,
            y_offset,
            x_offset + bar_width,
            y_offset + bar_height,
            outline="black",
            fill="lightgray",
            width=2,
        )

        # プログレス部分を描画
        if progress > 0:
            progress_width = int((progress / 100) * bar_width)
            canvas.create_rectangle(
                x_offset + 1,
                y_offset + 1,
                x_offset + progress_width - 1,
                y_offset + bar_height - 1,
                fill="blue",
                outline="",
            )

        # パーセンテージテキストを描画
        canvas.create_text(
            x_offset + bar_width // 2,
            y_offset + bar_height // 2,
            text=f"{progress}%",
            font=("Arial", 10),
            fill="black",
        )

    def create_layout(self):
        """GUIレイアウトを作成"""
        layout = [
            [sg.Text("マスター特徴量生成アプリ", font=("Arial", 16, "bold"))],
            [sg.HSeparator()],
            [sg.Text("モデルパラメータ設定", font=("Arial", 12, "bold"))],
            [sg.Text(f"設定ファイル: config/base_model_params.json", font=("Arial", 10))],
            [
                sg.Text(
                    f"Model Type: {self.model_params.get('model_type', 'N/A')}",
                    font=("Arial", 10),
                )
            ],
            [
                sg.Text(
                    f"Feature Layer: {self.model_params.get('feat_layer', 'N/A')}",
                    font=("Arial", 10),
                )
            ],
            [
                sg.Text(
                    f"Image Size: {self.model_params.get('image_size', 'N/A')}",
                    font=("Arial", 10),
                )
            ],
            [sg.HSeparator()],
            [sg.Text("処理対象製品選択", font=("Arial", 12, "bold"))],
            [
                sg.Button("全選択", key="-SELECT_ALL-", size=(8, 1)),
                sg.Button("全解除", key="-DESELECT_ALL-", size=(8, 1)),
            ],
            [
                sg.Listbox(
                    values=self.available_products,
                    select_mode="multiple",
                    size=(50, 8),
                    key="-PRODUCT_LIST-",
                )
            ],
            [sg.HSeparator()],
            [
                sg.Button(
                    "特徴量生成開始",
                    key="-START-",
                    size=(15, 2),
                    button_color=("white", "green"),
                ),
                sg.Button(
                    "停止",
                    key="-STOP-",
                    size=(15, 2),
                    button_color=("white", "red"),
                    disabled=True,
                ),
                sg.Button("終了", key="-EXIT-", size=(15, 2)),
            ],
            [sg.HSeparator()],
            [sg.Text("進捗:", font=("Arial", 10, "bold"))],
            [sg.Canvas(size=(500, 25), key="-PROGRESSBAR-", background_color="white")],
            [sg.Text("ステータス:", font=("Arial", 10, "bold"))],
            [sg.Text("", key="-STATUS-", size=(80, 1), font=("Arial", 10))],
        ]
        return layout

    def run(self):
        """アプリケーション実行"""
        # モデルパラメータ読み込み
        if not self.load_model_params():
            return

        # 利用可能製品一覧取得
        self.available_products = self.get_available_products()

        if not self.available_products:
            sg.popup_error("masterフォルダに製品が見つかりません")
            return

        layout = self.create_layout()

        window = sg.Window(
            "マスター特徴量生成アプリ",
            layout,
            finalize=True,
            resizable=True,
            grab_anywhere=False,
        )

        self.update_progress_bar(window, 0)
        window["-STATUS-"].update("アプリケーションを開始しました")

        while True:
            event, values = window.read(timeout=100)

            if event == sg.WIN_CLOSED or event == "-EXIT-":
                if self.is_processing:
                    self.is_processing = False
                break

            elif event == "-SELECT_ALL-":
                # tkinterのListboxウィジェットに直接アクセス
                listbox_widget = window["-PRODUCT_LIST-"].widget
                listbox_widget.selection_set(0, "end")  # 全選択

            elif event == "-DESELECT_ALL-":
                # tkinterのListboxウィジェットに直接アクセス
                listbox_widget = window["-PRODUCT_LIST-"].widget
                listbox_widget.selection_clear(0, "end")  # 全解除

            elif event == "-START-":
                # 選択された製品を取得
                selected_products = values["-PRODUCT_LIST-"]

                if not selected_products:
                    sg.popup_error("処理する製品を選択してください")
                    continue

                self.is_processing = True
                window["-START-"].update(disabled=True)
                window["-STOP-"].update(disabled=False)
                self.update_progress_bar(window, 0)

                window["-STATUS-"].update(f"選択された製品: {', '.join(selected_products)}")

                # 別スレッドで処理開始
                threading.Thread(
                    target=self.process_selected_products,
                    args=(selected_products, window),
                    daemon=True,
                ).start()

            elif event == "-STOP-":
                self.is_processing = False
                window["-START-"].update(disabled=False)
                window["-STOP-"].update(disabled=True)
                window["-STATUS-"].update("処理を停止しました")

            elif event == "-UPDATE-":
                message = (
                    values[event]
                    if isinstance(values[event], str)
                    else values[event].get("-UPDATE-", str(values[event]))
                )
                window["-STATUS-"].update(message)

            elif event == "-PROGRESS-":
                progress = (
                    values[event]
                    if isinstance(values[event], int)
                    else values[event].get("-PROGRESS-", 0)
                )
                self.update_progress_bar(window, progress)

            elif event == "-COMPLETE-":
                window["-START-"].update(disabled=False)
                window["-STOP-"].update(disabled=True)
                self.update_progress_bar(window, 100)
                message = (
                    values[event]
                    if isinstance(values[event], str)
                    else values[event].get("-COMPLETE-", str(values[event]))
                )
                window["-STATUS-"].update(message)
                sg.popup_ok("処理完了", message)

            elif event == "-ERROR-":
                window["-START-"].update(disabled=False)
                window["-STOP-"].update(disabled=True)
                message = (
                    values[event]
                    if isinstance(values[event], str)
                    else values[event].get("-ERROR-", str(values[event]))
                )
                window["-STATUS-"].update(f"エラー: {message}")
                sg.popup_error("処理エラー", message)

            elif event == "-ENABLE-":
                window["-START-"].update(disabled=False)
                window["-STOP-"].update(disabled=True)

        window.close()


def main():
    """エントリポイント"""
    try:
        app = MasterGeneratorApp()
        app.run()
    except Exception as e:
        sg.popup_error(f"アプリケーションエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
