# Segmentation/utils/visualization.py
"""
欠陥検出結果の可視化ユーティリティ
"""

from __future__ import annotations

import numpy as np
import cv2


def pad_to_square(image: np.ndarray, size: int) -> np.ndarray:
    """
    画像を正方形にパディング・リサイズ

    Args:
        image: 入力画像 (RGB)
        size: 出力サイズ

    Returns:
        np.ndarray: 正方形にリサイズされた画像
    """
    from PIL import Image, ImageOps

    pil = Image.fromarray(image)
    pil = ImageOps.pad(pil, (size, size), color=(0, 0, 0))
    return np.asarray(pil)


def to_png_bytes(image: np.ndarray) -> bytes:
    """
    画像をPNGバイト列に変換

    Args:
        image: RGB形式の画像

    Returns:
        bytes: PNGバイト列
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".png", image_bgr)[1].tobytes()
