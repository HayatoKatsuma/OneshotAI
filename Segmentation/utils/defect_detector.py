# Segmentation/utils/defect_detector.py
"""
SAM3の視覚プロンプト機能を使った欠陥検出クラス
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import sys
import io

import numpy as np
import torch
import cv2

from .defect_library import DefectLibrary

# SAM3モジュールへのパスを追加
_segmentation_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _segmentation_root not in sys.path:
    sys.path.insert(0, _segmentation_root)


def crop_horizontal_strip(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding_ratio: float = 0.1,
) -> np.ndarray:
    """
    BBOXの横方向のみで切り出し（縦方向は画像全体を保持）

    背景情報を保持するため、縦方向は元画像全体を使用し、
    横方向のみBBOXに基づいて切り出す。
    """
    h, w = image.shape[:2]
    cx, cy, bw, bh = bbox

    # 横方向のみパディング適用
    bw_padded = bw * (1 + padding_ratio)

    x1 = int((cx - bw_padded / 2) * w)
    x2 = int((cx + bw_padded / 2) * w)

    x1 = max(0, x1)
    x2 = min(w, x2)

    # 縦方向は画像全体を使用
    return image[:, x1:x2].copy()


@dataclass
class DetectionResult:
    """検出結果を格納するデータクラス"""
    total_defects: int
    defect_counts: Dict[str, int]  # 欠陥クラスごとのカウント {display_name: count}
    overlay_image: np.ndarray
    original_image: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_defects(self) -> bool:
        return self.total_defects > 0

    def get_summary(self) -> str:
        """欠陥クラスごとの内訳を文字列で返す"""
        if not self.defect_counts:
            return "検出なし"
        parts = [f"{name}: {count}個" for name, count in self.defect_counts.items() if count > 0]
        return ", ".join(parts) if parts else "検出なし"


class DefectDetector:
    """SAM3の視覚プロンプト機能を使った欠陥検出クラス"""

    def __init__(
        self,
        defect_library: DefectLibrary,
        device: str = "cuda",
        debug_save_dir: Optional[str] = None,
        nms_iou_threshold: float = 0.5,
        save_debug_images: bool = False,
    ):
        self.defect_library = defect_library
        self.device = device
        self.debug_save_dir = debug_save_dir
        self.nms_iou_threshold = nms_iou_threshold
        self.save_debug_images = save_debug_images

        if debug_save_dir and save_debug_images:
            os.makedirs(debug_save_dir, exist_ok=True)

        self._model = None
        self._processor = None  # Sam3Processorインスタンス（共有）
        self._model_loaded = False
        self._plot_results = None

    def preload_model(self) -> bool:
        """SAM3モデルを事前ロード"""
        if self._model_loaded:
            return True
        self._load_model()
        return True

    def _load_model(self) -> None:
        """SAM3モデルをロード"""
        if self._model_loaded:
            return

        print("SAM3モデルをロード中...")

        # TF32有効化
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.visualization_utils import plot_results

        bpe_path = os.path.join(_segmentation_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            bpe_path = None

        self._model = build_sam3_image_model(bpe_path=bpe_path, device=self.device)
        self._processor = Sam3Processor(self._model)  # 共有インスタンスを作成
        self._plot_results = plot_results
        self._model_loaded = True

        print("SAM3モデルのロード完了")

    @property
    def is_ready(self) -> bool:
        return self._model_loaded

    def _apply_nms(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        NMS（Non-Maximum Suppression）を適用して重複検出を除去

        Args:
            detections: 検出結果のリスト
            iou_threshold: IoU閾値（これ以上重なる検出は除去）

        Returns:
            NMS適用後の検出結果
        """
        if len(detections) == 0:
            return []

        # OpenCVのNMSBoxesはx,y,w,h形式を期待
        boxes_xywh = []
        scores = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            w = x2 - x1
            h = y2 - y1
            boxes_xywh.append([x1, y1, w, h])
            scores.append(det["score"])

        # OpenCVのNMSを使用
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh,
            scores=scores,
            score_threshold=0.0,  # 既にフィルタ済み
            nms_threshold=iou_threshold,
        )

        # OpenCV 4.x以降はindicesがflatten済み、またはリスト形式
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            return [detections[i] for i in indices]
        return []

    def _detect_with_visual_prompt(
        self,
        inspection_image: np.ndarray,
        reference_image: np.ndarray,
        reference_bbox: Tuple[float, float, float, float],
        confidence_threshold: float,
        defect_type_name: str,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        SAM3のVisual Promptで検出し、検出結果のリストを返す

        Args:
            inspection_image: 検査画像（RGB形式）
            reference_image: 参照画像（RGB形式、元画像）
            reference_bbox: 参照画像内の欠陥BBOX (cx, cy, w, h) 正規化座標
            confidence_threshold: 検出閾値
            defect_type_name: 欠陥タイプの表示名
            verbose: デバッグ出力

        Returns:
            List[Dict]: 検出結果のリスト [{"box": (x1,y1,x2,y2), "score": float, "type": str}, ...]
        """
        if not self._model_loaded:
            self._load_model()

        from PIL import Image as PILImage

        insp_h, insp_w = inspection_image.shape[:2]

        # 参照画像をBBOXの横方向のみで切り出し（縦は全体を保持）
        reference_strip = crop_horizontal_strip(
            reference_image, reference_bbox, padding_ratio=0.1
        )
        ref_h, ref_w = reference_strip.shape[:2]

        # 参照ストリップをリサイズ（検査画像と高さを揃える）
        scale = insp_h / ref_h
        new_ref_h = insp_h
        new_ref_w = int(ref_w * scale)
        ref_resized = cv2.resize(reference_strip, (new_ref_w, new_ref_h))

        if verbose:
            print(f"[DEBUG] 検査画像: {insp_w}x{insp_h}")
            print(f"[DEBUG] 参照ストリップ: {ref_w}x{ref_h} -> {new_ref_w}x{new_ref_h}")

        # 結合画像を作成（参照を左、検査を右）bine
        combined_w = new_ref_w + insp_w
        combined_h = insp_h
        combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        combined[:, :new_ref_w] = ref_resized
        combined[:, new_ref_w:] = inspection_image

        if verbose:
            print(f"[DEBUG] 結合画像: {combined_w}x{combined_h}")

        # 参照領域内の欠陥BBOXを結合画像座標に変換
        cx, cy, bw, bh = reference_bbox
        ref_cx = (new_ref_w / 2) / combined_w
        ref_cy = cy
        ref_bw = new_ref_w / combined_w
        ref_bh = bh
        norm_box = [ref_cx, ref_cy, ref_bw, ref_bh]

        if verbose:
            print(f"[DEBUG] 元BBOX: cx={cx:.3f}, cy={cy:.3f}, w={bw:.3f}, h={bh:.3f}")
            print(f"[DEBUG] 結合画像BBOX: cx={ref_cx:.3f}, cy={ref_cy:.3f}, w={ref_bw:.3f}, h={ref_bh:.3f}")

        # SAM3で処理
        combined_pil = PILImage.fromarray(combined)

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            # 各推論で新しいstateを作成（set_imageが新しいstateを返す）
            state = self._processor.set_image(combined_pil)
            # 閾値を設定
            self._processor.confidence_threshold = confidence_threshold
            # プロンプトをリセット
            self._processor.reset_all_prompts(state)
            state = self._processor.add_geometric_prompt(
                state=state,
                box=norm_box,
                label=True,
            )

        # 検出結果を収集（検査画像領域内のみ）
        detections = []
        if "boxes" in state and len(state["boxes"]) > 0:
            boxes = state["boxes"].cpu().numpy()
            scores = state["scores"].cpu().float().numpy()
            for box, score in zip(boxes, scores):
                box_center_x = (box[0] + box[2]) / 2
                if box_center_x >= new_ref_w and float(score) >= confidence_threshold:
                    # 検査画像座標に変換（参照画像幅を引く）
                    x1 = max(0, int(box[0]) - new_ref_w)
                    y1 = max(0, int(box[1]))
                    x2 = min(insp_w, int(box[2]) - new_ref_w)
                    y2 = min(insp_h, int(box[3]))
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "score": float(score),
                        "type": defect_type_name,
                    })

        if verbose:
            print(f"[DEBUG] 検出数（検査領域内）: {len(detections)}")

        # デバッグ保存（BBOX付き結合画像のみ）
        if self.save_debug_images and self.debug_save_dir:
            os.makedirs(self.debug_save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # BBOX付き結合画像を保存
            combined_with_bbox = combined.copy()
            # 正規化座標をピクセル座標に変換
            bbox_x1 = int((norm_box[0] - norm_box[2] / 2) * combined_w)
            bbox_y1 = int((norm_box[1] - norm_box[3] / 2) * combined_h)
            bbox_x2 = int((norm_box[0] + norm_box[2] / 2) * combined_w)
            bbox_y2 = int((norm_box[1] + norm_box[3] / 2) * combined_h)
            cv2.rectangle(combined_with_bbox, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 3)
            debug_bbox_path = os.path.join(self.debug_save_dir, f"combined_bbox_{defect_type_name}_{timestamp}.png")
            cv2.imwrite(debug_bbox_path, cv2.cvtColor(combined_with_bbox, cv2.COLOR_RGB2BGR))
            if verbose:
                print(f"[DEBUG] BBOX可視化画像保存: {debug_bbox_path}")

        return detections

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        defect_types: Dict,
    ) -> np.ndarray:
        """
        検出結果を画像に描画

        Args:
            image: 描画対象の画像（RGB形式）
            detections: 検出結果のリスト
            defect_types: 欠陥タイプの辞書（色情報取得用）

        Returns:
            np.ndarray: 描画後の画像
        """
        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            type_name = det["type"]

            # 欠陥タイプから色を取得（デフォルトは緑）
            color = (0, 255, 0)
            for dt in defect_types.values():
                if dt.display_name == type_name:
                    color = dt.color
                    break

            # バウンディングボックスを描画
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # ラベル（欠陥タイプと確信度）を描画
            label = f"{type_name}: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # テキストサイズを取得
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # 背景矩形を描画
            cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)

            # テキストを描画（白色）
            cv2.putText(result, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)

        return result

    def detect(self, image: np.ndarray, verbose: bool = True) -> DetectionResult:
        """画像から欠陥を検出（全欠陥タイプの結果をOR統合）"""
        if verbose:
            print(f"\n{'='*50}")
            print("欠陥検出開始（SAM3 Visual Prompt）")
            print(f"{'='*50}")

        all_detections = []  # 全欠陥タイプの検出結果を収集

        for defect_type in self.defect_library.defect_types.values():
            # 欠陥クラスごとにGPUキャッシュをクリア（前回の推論の影響を排除）
            if self.device == "cuda":
                torch.cuda.empty_cache()

            if verbose:
                print(f"\n[INFO] '{defect_type.display_name}' 処理中...")
                print(f"[INFO]   閾値: {defect_type.confidence_threshold}")

            reference_samples = self.defect_library.get_reference_samples_with_bbox(
                defect_type.name
            )
            if not reference_samples:
                if verbose:
                    print("[WARN]   参照画像なし")
                continue

            if verbose:
                print(f"[INFO]   参照サンプル数: {len(reference_samples)}")

            # 最初の参照サンプルで検出
            ref_image, ref_bbox = reference_samples[0]
            detections = self._detect_with_visual_prompt(
                inspection_image=image,
                reference_image=ref_image,
                reference_bbox=ref_bbox,
                confidence_threshold=defect_type.confidence_threshold,
                defect_type_name=defect_type.display_name,
                verbose=verbose,
            )

            # 欠陥タイプごとにNMSを適用
            detections_before_nms = len(detections)
            detections = self._apply_nms(detections, iou_threshold=self.nms_iou_threshold)
            all_detections.extend(detections)

            if verbose:
                if detections_before_nms != len(detections):
                    print(f"[NMS] {detections_before_nms} -> {len(detections)} (重複除去)")
                print(f"[RESULT] '{defect_type.display_name}': {len(detections)}個")

        # 欠陥クラスごとのカウントを計算
        defect_counts: Dict[str, int] = {}
        for det in all_detections:
            type_name = det["type"]
            defect_counts[type_name] = defect_counts.get(type_name, 0) + 1

        # 全検出結果を画像に描画
        total_defects = len(all_detections)
        result_image = self._draw_detections(
            image, all_detections, self.defect_library.defect_types
        )

        if verbose:
            print(f"\n{'='*50}")
            print(f"検出完了: 合計 {total_defects} 個")
            if defect_counts:
                for name, count in defect_counts.items():
                    print(f"  - {name}: {count}個")
            print(f"{'='*50}\n")

        # デバッグ保存（NG時のみ最終結果を保存）
        if self.save_debug_images and self.debug_save_dir and total_defects > 0:
            os.makedirs(self.debug_save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            result_path = os.path.join(self.debug_save_dir, f"result_{timestamp}.png")
            cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

        return DetectionResult(
            total_defects=total_defects,
            defect_counts=defect_counts,
            overlay_image=result_image,
            original_image=image,
        )

    def unload_model(self) -> None:
        """モデル解放"""
        if self._model is not None:
            del self._model
            self._model = None
            self._processor = None
            self._model_loaded = False
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print("SAM3モデルを解放")
