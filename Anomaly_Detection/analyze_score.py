#!/usr/bin/env python3
"""
analyze_score.py - 異常度計算と時系列プロット生成

各製品のmaster.npyファイルを使用して、--target_dirで指定されたディレクトリ以下にある
全BMPファイルの異常度を計算し、ファイル名の数字順に時系列プロットを生成します。
"""

import os
import sys
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import cv2
import torch
import faiss
import argparse


# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backbones import get_model
from src.utils import augment_image


def configure_matplotlib_font():
    """Configure matplotlib for English output"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("Using DejaVu Sans font for English labels")


class AnomalyScoreCalculator:
    """異常度計算クラス"""
    
    def __init__(self, model_type: str, feat_layer: int, image_size: int, roi_dino: Tuple[float, float, float, float]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.feat_layer = feat_layer
        self.image_size = image_size
        self.model = None
        self.roi_dino = roi_dino
        self.current_master_feat = None
        self.current_knn = None
        self.current_master_hue_deg: float = None  # 追加：マスター側の平均Hue（度）
        self._init_model()
    
    def _init_model(self):
        """モデルの初期化"""
        self.model = get_model(
            self.model_type,
            self.device,
            smaller_edge_size=self.image_size,
            feat_layer=self.feat_layer,
        )
    
    def create_master_features(self, master_folder_path: str, masking: bool = False, rotation: bool = False):
        """マスター画像の特徴量ファイルを作成（既存ロジックを維持）"""
        master_folder = Path(master_folder_path)
        model_name = self.model_type.replace('dinov2_', '')
        npy_filename = f"master_{model_name}_{self.feat_layer}_{self.image_size}.npy"
        npy_path = master_folder / npy_filename
        
        master_image_paths = [p for p in master_folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.bmp'}]
        if not master_image_paths:
            print(f"No BMP files found in {master_folder}")
            return None
        
        print(f"Creating features for {len(master_image_paths)} master images in {master_folder}")

        all_features = []
        for master_image_path in master_image_paths:
            # 重要：DINO用はRGB
            master_image = self._crop_roi_rel(cv2.cvtColor(cv2.imread(str(master_image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), self.roi_dino)
            if rotation:
                img_augmented = augment_image(master_image)
                for img in img_augmented:
                    master_tensor, _ = self.model.prepare_image(img)
                    feature = self.model.extract_features(master_tensor)
                    all_features.append(feature)
            else:
                master_tensor, _ = self.model.prepare_image(master_image)
                feature = self.model.extract_features(master_tensor)
                all_features.append(feature)
        
        if all_features:
            concat_feature = np.concatenate(all_features, axis=0)
            np.save(npy_path, concat_feature)
            print(f"Successfully saved features to: {npy_path}")
            return str(npy_path)
        
        return None
    
    def load_master_features(self, master_npy_path: str):
        """マスター特徴量の読み込み"""
        self.current_master_feat = np.load(master_npy_path)
        faiss.normalize_L2(self.current_master_feat)
        self.current_knn = faiss.IndexFlatL2(self.current_master_feat.shape[1])
        self.current_knn.add(self.current_master_feat)

    # ============ Hue 関連（保存・計算・スコア化） ============
    @staticmethod
    def _crop_roi_rel(img_bgr: np.ndarray, roi_rel: Tuple[float, float, float, float]) -> np.ndarray:
        """相対ROIで矩形切り出し（BGR前提）"""
        h, w = img_bgr.shape[:2]
        y0 = int(roi_rel[0] * h); y1 = int(roi_rel[1] * h)
        x0 = int(roi_rel[2] * w); x1 = int(roi_rel[3] * w)
        return img_bgr[y0:y1, x0:x1]

    @staticmethod
    def _circular_mean_deg(hue_deg_1d: np.ndarray) -> float:
        """角度データの円環平均（度）。"""
        rad = np.deg2rad(hue_deg_1d)
        c = np.cos(rad).mean()
        s = np.sin(rad).mean()
        if c == 0 and s == 0:
            return float('nan')
        return float((math.degrees(math.atan2(s, c)) % 360.0))

    @staticmethod
    def _angular_distance_deg(a: float, b: float) -> float:
        """最小角距離（度, 0..180）"""
        diff = abs(a - b) % 360.0
        return min(diff, 360.0 - diff)

    def _compute_master_hue_from_folder(
        self,
        master_folder_path: str,
        roi_rel: Tuple[float, float, float, float],
        sat_thresh: int
    ) -> float:
        """マスター側の平均Hue（度）を、マスター内の全BMPから平均して求める"""
        master_folder = Path(master_folder_path)
        bmp_paths = [p for p in master_folder.rglob("*.bmp")]
        if not bmp_paths:
            print(f"[WARN] No BMP found in master folder for Hue computation: {master_folder}")
            return float('nan')

        means = []
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
                means.append(self._circular_mean_deg(hue[mask]))
        if not means:
            print(f"[WARN] Failed to compute master hue (no valid pixels): {master_folder}")
            return float('nan')
        return float(np.mean(means))

    def ensure_master_hue_file(
        self,
        master_folder_path: str,
        roi_rel: Tuple[float, float, float, float],
        sat_thresh: int
    ) -> float:
        """
        master_hue.txt を master/<製品フォルダ>/ に毎回新しく作成・保存する。
        値は平均Hue（度, 0..360）。
        """
        folder = Path(master_folder_path)
        txt_path = folder / "master_hue.txt"
        
        val = self._compute_master_hue_from_folder(master_folder_path, roi_rel, sat_thresh)
        self.current_master_hue_deg = val
        try:
            txt_path.write_text(f"{val:.6f}\n", encoding="utf-8")
            print(f"Created master_hue.txt: {txt_path} (value={val:.6f} deg)")
        except Exception as e:
            print(f"[WARN] Failed to write master_hue.txt: {e}")
        return val

    def calculate_hue_score(
        self,
        image_bgr: np.ndarray,
        master_hue_deg: float,
        roi_rel: Tuple[float, float, float, float],
        sat_thresh: int,
        debug: bool = False  # ★追加：確認ログ用
    ) -> float:
        """
        平均Hue異常度（1 - cos 距離を最大1.0に正規化）を返す。
        - 入力はBGR配列（cv2.imreadのまま）
        - HueはBGR→HSV（OpenCV Hueは0..179 → ×2で0..358）
        - ★実験どおり：S>sat_thresh でフィルタ後の Hue 円環平均を使用
        """
        if image_bgr is None:
            return 0.0

        roi = self._crop_roi_rel(image_bgr, roi_rel)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float32) * 2.0
        sat = hsv[:, :, 1].astype(np.float32)

        mask = sat > float(sat_thresh)
        if not np.any(mask):
            if debug:
                print(f"[HueDebug] No valid pixels after S>{sat_thresh} in ROI={roi_rel}")
            return 0.0

        mean_hue_deg = self._circular_mean_deg(hue[mask])

        if debug:
            valid_px = int(mask.sum())
            roi_h, roi_w = roi.shape[:2]
            print(f"[HueDebug] ROI={roi_rel} size={roi_w}x{roi_h}, S>{sat_thresh} valid_px={valid_px}, "
                  f"circular_mean_hue={mean_hue_deg:.2f}°")

        dtheta = self._angular_distance_deg(mean_hue_deg, master_hue_deg)
        d = 1.0 - math.cos(math.radians(dtheta))  # 0..2
        d_norm = min(d / 2.0, 1.0)                # 最大1.0に規格化
        return float(d_norm)
    # =====================================

    @torch.no_grad()
    def calculate_anomaly_score(self, image_bgr: np.ndarray) -> float:
        """
        単一画像の異常度計算（AnomalyDINO）
        - 入力はBGR配列（cv2.imreadのまま）
        - 内部でRGBへ変換
        """
        if image_bgr is None:
            return float('inf')
        image_rgb = self._crop_roi_rel(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), self.roi_dino)
        tensor, _ = self.model.prepare_image(image_rgb)
        feat = self.model.extract_features(tensor)
        faiss.normalize_L2(feat)
        dist, _ = self.current_knn.search(feat, 1)
        dist = dist / 2.0
        return float(np.mean(sorted(dist, reverse=True)[:max(1, int(len(dist) * 0.01))]))


def get_bmp_files_recursive(target_dir: str) -> List[str]:
    """指定ディレクトリ以下の全.bmpファイルを再帰的に取得"""
    target_path = Path(target_dir)
    if not target_path.exists():
        print(f"Directory does not exist: {target_dir}")
        return []
    bmp_files = list(target_path.rglob("*.bmp"))
    bmp_files = [str(f) for f in bmp_files]

    def extract_number(filename):
        import re
        match = re.search(r'(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else 0
    bmp_files.sort(key=extract_number)
    return bmp_files


def get_subfolders_with_bmp(target_dir: str) -> List[str]:
    """指定ディレクトリ内でBMPファイルを含むサブフォルダを取得"""
    target_path = Path(target_dir)
    if not target_path.exists():
        return []
    subfolders = []
    for item in target_path.iterdir():
        if item.is_dir() and list(item.glob("*.bmp")):
            subfolders.append(item.name)
    return sorted(subfolders)


def has_direct_bmp_files(target_dir: str) -> bool:
    """指定ディレクトリに直接BMPファイルがあるかチェック"""
    target_path = Path(target_dir)
    return bool(list(target_path.glob("*.bmp")))


def get_master_folders() -> List[str]:
    """マスターフォルダの一覧を取得"""
    master_dir = Path("master")
    master_folders = []
    for folder in master_dir.iterdir():
        if folder.is_dir() and list(folder.glob("*.bmp")):
            master_folders.append(folder.name)
    return sorted(master_folders)


def calculate_scores_for_directory(calculator: AnomalyScoreCalculator, 
                                   master_folder: str, 
                                   target_directory: str,
                                   roi_rel: Tuple[float, float, float, float],
                                   sat_thresh: int,
                                   hue_weight: float,
                                   output_dir: Path = None) -> Tuple[List[float], List[str], float, np.ndarray, int]:
    """指定ディレクトリ以下の全.bmpファイルに対する異常度計算"""
    # マスター特徴量ファイルの作成（存在しない場合）
    master_folder_path = f"master/{master_folder}"
    npy_path = calculator.create_master_features(master_folder_path)
    if npy_path is None:
        raise Exception(f"Failed to create or find master features for {master_folder}")
    
    # マスター特徴量の読み込み
    calculator.load_master_features(npy_path)

    # master_hue.txt を確保（無ければ計算して保存／有れば読み込み）
    master_hue = calculator.ensure_master_hue_file(master_folder_path, roi_rel, sat_thresh)
    print(f"  Master mean Hue (deg): {master_hue:.2f}")

    # BMPファイルの一覧取得（再帰的）
    bmp_files = get_bmp_files_recursive(target_directory)
    if not bmp_files:
        print(f"No BMP files found in {target_directory}")
        return [], [], 0.0, None, -1
    
    dino_scores = []
    filenames = []
    min_dino_score = float('inf')
    min_dino_score_image = None
    min_dino_score_index = -1
    print(f"Processing {len(bmp_files)} images for {master_folder} in {target_directory}")

    for i, bmp_file in enumerate(bmp_files):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(bmp_files)}")
        
        # ---- imread は1回だけ ----
        img_bgr = cv2.imread(bmp_file, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # 読み込み失敗時は極端なスコアにしておく
            dino_scores.append(float('inf'))
            filenames.append(os.path.relpath(bmp_file, target_directory))
            continue

        # AnomalyDINO 異常度（BGR→RGB内で変換）
        dino_score = calculator.calculate_anomaly_score(img_bgr)
        dino_scores.append(dino_score)
        filenames.append(os.path.relpath(bmp_file, target_directory))
        
        # 最小dino_scoreの画像を追跡
        if dino_score < min_dino_score and dino_score != float('inf'):
            min_dino_score = dino_score
            min_dino_score_image = img_bgr.copy()
            min_dino_score_index = i
    
    # 最小dino_scoreの画像に対してhue_scoreを計算し、final_scoreを算出して保存
    if min_dino_score_image is not None and output_dir is not None:
        # master_hueがNaNの場合はhue_scoreを0にする
        if np.isnan(master_hue):
            hue_score = 0.0
        else:
            hue_score = calculator.calculate_hue_score(
                image_bgr=min_dino_score_image,
                master_hue_deg=master_hue,
                roi_rel=roi_rel,
                sat_thresh=sat_thresh,
                debug=True
            )
        
        final_score = min_dino_score + hue_weight * hue_score
        product_name = os.path.basename(target_directory.rstrip('/'))
        save_min_score_image_with_roi(min_dino_score_image, master_folder, product_name, 
                                      min_dino_score_index, roi_rel, output_dir, final_score)
    
    return dino_scores, filenames, min_dino_score, min_dino_score_image, min_dino_score_index


def save_min_score_image_with_roi(image_bgr: np.ndarray, 
                                   master_name: str, 
                                   product_name: str, 
                                   index: int,
                                   roi_rel: Tuple[float, float, float, float],
                                   output_dir: Path,
                                   final_score: float) -> None:
    """最小dino_score画像をROI描画とfinal_score表示付きで保存"""
    # ROIの描画
    h, w = image_bgr.shape[:2]
    y0 = int(roi_rel[0] * h)
    y1 = int(roi_rel[1] * h)
    x0 = int(roi_rel[2] * w)
    x1 = int(roi_rel[3] * w)
    
    # 画像をコピーしてROI矩形を描画
    img_with_roi = image_bgr.copy()
    cv2.rectangle(img_with_roi, (x0, y0), (x1, y1), (0, 255, 0), 2)  # 緑色の矩形
    
    # final_scoreをテキストで描画
    score_text = f"Final Score: {final_score:.4f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (0, 255, 0)  # 緑色
    thickness = 2
    
    # テキストサイズを取得して背景矩形を描画
    text_size = cv2.getTextSize(score_text, font, font_scale, thickness)[0]
    text_x = 10
    text_y = 40
    cv2.rectangle(img_with_roi, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)  # 黒い背景
    cv2.putText(img_with_roi, score_text, (text_x, text_y), font, font_scale, color, thickness)
    
    # ファイル名の生成
    filename = f"{master_name}_{product_name}_{index}.bmp"
    output_path = output_dir / filename
    
    # 画像の保存
    cv2.imwrite(str(output_path), img_with_roi)
    print(f"Saved minimum dino_score image with ROI and final_score: {filename}")


def create_timeseries_plot(scores: List[float], 
                          filenames: List[str],
                          master_name: str, 
                          product_name: str, 
                          output_dir: Path) -> None:
    """時系列プロットの作成と保存（dino_scoreをプロット）"""
    if not scores:
        return
    
    max_score = np.max(scores)
    min_score = np.min(scores)
    median_score = np.median(scores)
    mean_score = np.mean(scores)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(scores)), scores, 'b-', linewidth=1, alpha=0.7)
    plt.scatter(range(len(scores)), scores, c='red', s=10, alpha=0.6)
    plt.xlabel('Image Index (sorted by filename number)')
    plt.ylabel('DINO Score')
    plt.title(f'DINO Score Time Series: {master_name} -> {product_name}')
    plt.grid(True, alpha=0.3)
    
    stats_text = f"""Statistics:
Max: {max_score:.4f}
Min: {min_score:.4f}
Median: {median_score:.4f}
Mean: {mean_score:.4f}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    filename = f"{master_name}_{product_name}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved time series plot: {filename}")
    print(f"  Max: {max_score:.4f}, Min: {min_score:.4f}, Median: {median_score:.4f}, Mean: {mean_score:.4f}")


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='異常度計算と時系列プロット生成')
    parser.add_argument('--model_type', type=str, default='dinov2_vits14',
                        help='使用するモデルタイプ (例: dinov2_vits14, dinov2_vitb14)')
    parser.add_argument('--feat_layer', type=int, default=5,
                        help='特徴量抽出層 (DINOv2モデルのみ)')
    parser.add_argument('--image_size', type=int, default=504,
                        help='画像サイズ')
    parser.add_argument('--target_dir', type=str, default='data/2025_09_05', help='処理対象ディレクトリ（指定ディレクトリ以下の.bmpファイルを再帰的に処理）')
    parser.add_argument('--masters', type=str, nargs='*', default=['NKR', 'ファーストキッチン', 'マルガク', '特級', '標準', '無地'],
                        help='処理するマスター画像のフォルダ名（複数指定可能）')
    # ROI1=(0.20, 0.45, 0.25, 0.60)  ROI2=(0.70, 0.93, 0.11, 0.39)  ROI3=(0.70, 1.0, 0.0, 0.39)
    parser.add_argument('--roi_rel', type=float, nargs=4, default=[0.70, 0.93, 0.11, 0.39],
                        help='ROI相対座標 (y_start, y_end, x_start, x_end) default: 0.20 0.45 0.25 0.60')
    parser.add_argument('--roi_dino', type=float, nargs=4, default=[0.0, 1.0, 0.0, 1.0],
                        help='ROI相対座標 (y_start, y_end, x_start, x_end) default: 0.0 1.0 0.0 1.0')
    # ★S のデフォルトしきい値を 40 に変更（実験どおり）
    parser.add_argument('--sat_thresh', type=int, default=40,
                        help='彩度しきい値（低彩度はHue不安定のため除外） default: 40')
    parser.add_argument('--hue_weight', type=float, default=0.10,
                        help='最終スコアでのHue重み（係数） default: 0.2')
    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()
    print(f"Parameters: model_type={args.model_type}, feat_layer={args.feat_layer}, image_size={args.image_size}")
    print(f"ROI relative: {tuple(args.roi_rel)}, SAT_THRESH: {args.sat_thresh}, HUE_WEIGHT: {args.hue_weight}")
    print(f"Target directory: {args.target_dir}")
    configure_matplotlib_font()
    
    calculator = AnomalyScoreCalculator(
        model_type=args.model_type,
        feat_layer=args.feat_layer,
        image_size=args.image_size,
        roi_dino = args.roi_dino
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = calculator.model_type.replace('dinov2_', '')
    roi_rel_str = "_".join([f"{x:.2f}" for x in args.roi_rel])
    output_folder_name = f"{timestamp}_{model_name}_{calculator.feat_layer}_{calculator.image_size}_{roi_rel_str}"
    output_dir = Path("analyze") / output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    all_master_folders = get_master_folders()
    
    if args.masters:
        master_folders = []
        for master in args.masters:
            if master in all_master_folders:
                master_folders.append(master)
            else:
                print(f"Warning: Master folder '{master}' not found in available masters: {all_master_folders}")
        if not master_folders:
            print("Error: No valid master folders specified")
            return
        print(f"Processing selected {len(master_folders)} master folders: {master_folders}")
    else:
        master_folders = all_master_folders
        print(f"Processing all {len(master_folders)} master folders: {master_folders}")
    
    subfolders = get_subfolders_with_bmp(args.target_dir)
    has_direct_bmps = has_direct_bmp_files(args.target_dir)
    
    if subfolders and not has_direct_bmps:
        print(f"Found subfolders with BMP files: {subfolders}")
        print(f"Processing each subfolder separately in: {args.target_dir}")
        for subfolder in subfolders:
            subfolder_path = os.path.join(args.target_dir, subfolder)
            print(f"\n=== Processing subfolder: {subfolder} ===")
            for master_folder in master_folders:
                print(f"\nProcessing with master: {master_folder} -> {subfolder}")
                try:
                    dino_scores, filenames, min_dino_score, min_dino_score_image, min_dino_score_index = calculate_scores_for_directory(
                        calculator, master_folder, subfolder_path,
                        roi_rel=tuple(args.roi_rel), sat_thresh=args.sat_thresh, hue_weight=args.hue_weight,
                        output_dir=output_dir
                    )
                    if dino_scores:
                        create_timeseries_plot(dino_scores, filenames, master_folder, subfolder, output_dir)
                    else:
                        print(f"No scores calculated for {master_folder} -> {subfolder}")
                except Exception as e:
                    print(f"Error processing {master_folder} -> {subfolder}: {e}")
                    continue
    else:
        print(f"Processing all .bmp files in: {args.target_dir}")
        for master_folder in master_folders:
            print(f"\nProcessing with master: {master_folder}")
            try:
                dino_scores, filenames, min_dino_score, min_dino_score_image, min_dino_score_index = calculate_scores_for_directory(
                    calculator, master_folder, args.target_dir,
                    roi_rel=tuple(args.roi_rel), sat_thresh=args.sat_thresh, hue_weight=args.hue_weight,
                    output_dir=output_dir
                )
                if dino_scores:
                    target_name = os.path.basename(args.target_dir.rstrip('/'))
                    create_timeseries_plot(dino_scores, filenames, master_folder, target_name, output_dir)
                else:
                    print(f"No scores calculated for {master_folder}")
            except Exception as e:
                print(f"Error processing {master_folder}: {e}")
                continue
    
    print(f"\nCompleted! All time series plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
