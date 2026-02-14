# utils/anomaly_dino_detector.py
"""
AnomalyDINODetector - DINOv2 推論ラッパークラス
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import math

import numpy as np
import torch
import faiss
import cv2
from scipy.ndimage import gaussian_filter

from Anomaly_Detection.src.backbones import get_model


class AnomalyDINODetector:
    """
    ・torch-hub の dinov2_vit* checkpoints をそのまま使う
    ・k-NN (faiss) で 1-クラス判定し最小スコアを返す
    """

    def __init__(
        self,
        model_type: str,
        feat_layer: int,
        master_feat_path: Path,
        image_size: int,
        threshold: float,
        roi_rel: Tuple[float, float, float, float],
        sat_thresh: int,
        hue_weight: float,
        product_name: str,
    ) -> None:

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # backbone load
        self._model = get_model(
            model_type,
            self._device,
            smaller_edge_size=image_size,
            feat_layer=feat_layer,
        )

        # master feature
        self._master_feat: np.ndarray = np.load(master_feat_path)          # shape (N, D)

        # K-NN index（GPU利用可能ならGPU版を使用、失敗時はCPUにフォールバック）
        faiss.normalize_L2(self._master_feat)
        dim = self._master_feat.shape[1]
        cpu_index = faiss.IndexFlatL2(dim)
        cpu_index.add(self._master_feat)

        # GPUが利用可能かチェックしてGPUインデックスに変換
        if faiss.get_num_gpus() > 0:
            try:
                self._gpu_res = faiss.StandardGpuResources()
                self._knn = faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
                # GPU動作確認のためダミー検索を実行
                dummy = self._master_feat[:1].copy()
                self._knn.search(dummy, 1)
                print(f"FAISS: GPUインデックスを使用 (GPU 0)")
            except Exception as e:
                print(f"FAISS: GPU初期化失敗 ({e}), CPUにフォールバック")
                self._knn = cpu_index
        else:
            self._knn = cpu_index
            print("FAISS: CPUインデックスを使用")

        self._threshold = threshold
        self._roi_rel = roi_rel
        self._sat_thresh = sat_thresh
        self._hue_weight = hue_weight
        self._product_name = product_name
        
        # Load master hue value
        self._master_hue_deg = self._load_master_hue()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _score_single(
        self, frame: np.ndarray
    ) -> Tuple[float, np.ndarray, Tuple[int, int]]:
        """単一フレームの異常スコア、パッチ距離、グリッドサイズを返す"""
        # フレームはBaumerCameraクラスでRGB変換済み
        tensor, grid_size = self._model.prepare_image(frame)
        feat = self._model.extract_features(tensor)
        faiss.normalize_L2(feat)
        dist, _ = self._knn.search(feat, 1)          # 最近傍距離
        dist = dist / 2.0
        score = float(np.mean(sorted(dist.flatten(), reverse=True)[: max(1, int(len(dist) * 0.01))]))
        return score, dist.reshape(grid_size), grid_size

    # ------------------------------------------------------------------ #
    def _load_master_hue(self) -> float:
        """Load master hue value from master_hue.txt file"""
        if not self._product_name:
            return float('nan')
            
        master_hue_path = Path(f"Anomaly_Detection/master/{self._product_name}/master_hue.txt")
        if master_hue_path.exists():
            try:
                hue_value = float(master_hue_path.read_text().strip())
                return hue_value
            except (ValueError, OSError):
                return float('nan')
        return float('nan')
        
    @staticmethod
    def _crop_roi_rel(img_bgr: np.ndarray, roi_rel: Tuple[float, float, float, float]) -> np.ndarray:
        """ROI切り出し（BGR前提）"""
        h, w = img_bgr.shape[:2]
        y0 = int(roi_rel[0] * h); y1 = int(roi_rel[1] * h)
        x0 = int(roi_rel[2] * w); x1 = int(roi_rel[3] * w)
        return img_bgr[y0:y1, x0:x1]
        
    @staticmethod
    def _circular_mean_deg(hue_deg_1d: np.ndarray) -> float:
        """角度データの円環平均（度）"""
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
        
    def _calculate_hue_score(self, image_rgb: np.ndarray, master_hue_deg: float) -> float:
        """平均Hue異常度を計算"""
        if image_rgb is None or np.isnan(master_hue_deg):
            return 0.0
            
        # RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        roi = self._crop_roi_rel(image_bgr, self._roi_rel)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float32) * 2.0  # 0..360相当
        sat = hsv[:, :, 1].astype(np.float32)

        mask = sat > float(self._sat_thresh)
        if not np.any(mask):
            return 0.0

        mean_hue_deg = self._circular_mean_deg(hue[mask])
        dtheta = self._angular_distance_deg(mean_hue_deg, master_hue_deg)
        d = 1.0 - math.cos(math.radians(dtheta))  # 0..2
        d_norm = min(d / 2.0, 1.0)  # 最大1.0に規格化
        return float(d_norm)

    def _create_heatmap_overlay(
        self,
        image: np.ndarray,
        patch_dists: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """パッチ距離からヒートマップオーバーレイ画像を生成"""
        h, w = image.shape[:2]
        # パッチ距離を画像サイズにリサイズ
        heatmap = cv2.resize(patch_dists.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = gaussian_filter(heatmap, sigma=4)
        # 正規化 (0-1)
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        if heatmap_max - heatmap_min > 1e-8:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = np.zeros_like(heatmap)
        # カラーマップ適用 (JET: 青→緑→赤)
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        # オーバーレイ
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    def inspect_min(
        self, frames: List[np.ndarray]
    ) -> Tuple[float, np.ndarray, np.ndarray, str]:
        """フレーム列の最小DINOスコアと、そのフレーム、ヒートマップ付き画像、OK/NG を返却"""
        best_dino = float("inf")
        best_frame: np.ndarray | None = None
        best_patch_dists: np.ndarray | None = None

        # 1. 最小DINOスコアのフレームを探す
        for f in frames:
            score, patch_dists, _ = self._score_single(f)
            if score < best_dino:
                best_dino = score
                best_frame = f
                best_patch_dists = patch_dists

        if best_frame is None or best_patch_dists is None:
            dummy = np.zeros((100, 100, 3), np.uint8)
            return float("inf"), dummy, dummy, "NG"

        # 2. ヒートマップ画像を生成
        heatmap_image = self._create_heatmap_overlay(best_frame, best_patch_dists)

        # 3. Hueスコアを計算
        hue_score = self._calculate_hue_score(best_frame, self._master_hue_deg)

        # 4. 最終スコア: min_dino_score + hue_weight * hue_score
        final_score = best_dino + self._hue_weight * hue_score

        return final_score, best_frame, heatmap_image, ("OK" if final_score < self._threshold else "NG")