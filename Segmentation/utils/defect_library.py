# Segmentation/utils/defect_library.py
"""
欠陥サンプルを管理するライブラリクラス

SAM3の視覚プロンプト用にクロップ画像を保存・管理する。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import uuid
import shutil

import numpy as np
import cv2


@dataclass
class DefectSample:
    """個別の欠陥サンプル"""
    sample_id: str
    image_path: Path  # クロップ画像のパス
    bbox: Tuple[float, float, float, float]  # (cx, cy, w, h) 正規化座標
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """JSON保存用の辞書に変換"""
        return {
            "sample_id": self.sample_id,
            "image_path": str(self.image_path.name),
            "bbox": list(self.bbox),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict, samples_dir: Path) -> "DefectSample":
        """辞書からインスタンスを生成"""
        return cls(
            sample_id=data["sample_id"],
            image_path=samples_dir / data["image_path"],
            bbox=tuple(data["bbox"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class DefectType:
    """欠陥カテゴリ定義"""
    name: str
    display_name: str  # 日本語表示名
    color: Tuple[int, int, int]  # RGB表示色
    confidence_threshold: float  # 検出閾値（0-1）
    size_tolerance: float  # サイズ許容範囲（0-1、例：0.3 = ±30%）
    samples: List[DefectSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON保存用の辞書に変換"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "color": list(self.color),
            "confidence_threshold": self.confidence_threshold,
            "size_tolerance": self.size_tolerance,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict, type_dir: Path) -> "DefectType":
        """辞書からインスタンスを生成"""
        samples_dir = type_dir / "samples"
        samples = []
        for s_data in data.get("samples", []):
            samples.append(DefectSample.from_dict(s_data, samples_dir))

        # 後方互換性: min_areaがある場合はsize_toleranceに変換（デフォルト0.3）
        size_tolerance = data.get("size_tolerance", 0.3)
        if "min_area" in data and "size_tolerance" not in data:
            size_tolerance = 0.3  # 旧形式の場合はデフォルト値を使用

        return cls(
            name=data["name"],
            display_name=data["display_name"],
            color=tuple(data["color"]),
            confidence_threshold=data["confidence_threshold"],
            size_tolerance=size_tolerance,
            samples=samples,
        )


class DefectLibrary:
    """
    欠陥サンプルライブラリを管理するクラス

    ディレクトリ構造:
    defect_library/
    └── library_name/
        ├── scratch/
        │   ├── defect_config.json
        │   └── samples/
        │       ├── sample_001_crop.png
        │       └── ...
        └── hole/
            ├── defect_config.json
            └── samples/
    """

    def __init__(self, library_path: Path):
        """
        Args:
            library_path: ライブラリのルートディレクトリ
        """
        self.library_path = Path(library_path)
        self.defect_types: Dict[str, DefectType] = {}

    def load(self) -> bool:
        """
        ライブラリを読み込み

        Returns:
            bool: 読み込み成功時True
        """
        if not self.library_path.exists():
            print(f"ライブラリが存在しません: {self.library_path}")
            return False

        self.defect_types.clear()

        # 各欠陥タイプディレクトリを走査
        for type_dir in self.library_path.iterdir():
            if not type_dir.is_dir():
                continue

            config_path = type_dir / "defect_config.json"
            if not config_path.exists():
                continue

            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            defect_type = DefectType.from_dict(data, type_dir)
            self.defect_types[defect_type.name] = defect_type

        return True

    def save(self) -> bool:
        """
        ライブラリを保存

        Returns:
            bool: 保存成功時True
        """
        try:
            self.library_path.mkdir(parents=True, exist_ok=True)

            for defect_type in self.defect_types.values():
                type_dir = self.library_path / defect_type.name
                type_dir.mkdir(exist_ok=True)

                # サンプルディレクトリ作成
                samples_dir = type_dir / "samples"
                samples_dir.mkdir(exist_ok=True)

                # defect_config.json保存
                config_path = type_dir / "defect_config.json"
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(defect_type.to_dict(), f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"ライブラリの保存エラー: {e}")
            return False

    def add_defect_type(
        self,
        name: str,
        display_name: str,
        color: Tuple[int, int, int],
        confidence_threshold: float = 0.5,
        size_tolerance: float = 0.3,
    ) -> DefectType:
        """
        新しい欠陥カテゴリを追加

        Args:
            name: カテゴリ名（英字）
            display_name: 表示名（日本語可）
            color: 表示色 (R, G, B)
            confidence_threshold: 検出閾値
            size_tolerance: サイズ許容範囲（0-1、例：0.3 = ±30%）

        Returns:
            DefectType: 作成された欠陥タイプ
        """
        defect_type = DefectType(
            name=name,
            display_name=display_name,
            color=color,
            confidence_threshold=confidence_threshold,
            size_tolerance=size_tolerance,
            samples=[],
        )
        self.defect_types[name] = defect_type
        return defect_type

    def remove_defect_type(self, name: str) -> bool:
        """
        欠陥カテゴリを削除

        Args:
            name: カテゴリ名

        Returns:
            bool: 削除成功時True
        """
        if name not in self.defect_types:
            return False

        # ファイルシステムからも削除
        type_dir = self.library_path / name
        if type_dir.exists():
            shutil.rmtree(type_dir)

        del self.defect_types[name]
        return True

    def add_sample(
        self,
        defect_type_name: str,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        cropped_image: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        欠陥サンプルを追加

        Args:
            defect_type_name: 欠陥カテゴリ名
            image: 元のサンプル画像（RGB形式）- cropped_imageがない場合に使用
            bbox: BBOX座標 (cx, cy, w, h) 正規化座標
            cropped_image: クロップ済み画像（RGB形式）- 視覚プロンプト用

        Returns:
            str | None: 追加されたサンプルID、失敗時None
        """
        if defect_type_name not in self.defect_types:
            print(f"欠陥タイプが存在しません: {defect_type_name}")
            return None

        defect_type = self.defect_types[defect_type_name]

        # サンプルID生成
        sample_id = str(uuid.uuid4())[:8]

        # サンプル画像保存ディレクトリ
        samples_dir = self.library_path / defect_type_name / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # クロップ画像を保存
        if cropped_image is not None:
            image_path = samples_dir / f"{sample_id}_crop.png"
            crop_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), crop_bgr)
        else:
            # クロップ画像がない場合は元画像を保存
            image_path = samples_dir / f"{sample_id}.png"
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), image_bgr)

        # サンプル作成
        sample = DefectSample(
            sample_id=sample_id,
            image_path=image_path,
            bbox=bbox,
            created_at=datetime.now(),
        )

        defect_type.samples.append(sample)
        return sample_id

    def remove_sample(self, defect_type_name: str, sample_id: str) -> bool:
        """
        欠陥サンプルを削除

        Args:
            defect_type_name: 欠陥カテゴリ名
            sample_id: サンプルID

        Returns:
            bool: 削除成功時True
        """
        if defect_type_name not in self.defect_types:
            return False

        defect_type = self.defect_types[defect_type_name]

        for i, sample in enumerate(defect_type.samples):
            if sample.sample_id == sample_id:
                # 画像ファイル削除
                if sample.image_path.exists():
                    sample.image_path.unlink()

                defect_type.samples.pop(i)
                return True

        return False

    def get_reference_crops(self, defect_type_name: str) -> List[np.ndarray]:
        """
        カテゴリの全サンプルのクロップ画像を取得（視覚プロンプト用）

        Args:
            defect_type_name: 欠陥カテゴリ名

        Returns:
            List[np.ndarray]: クロップ画像のリスト (RGB形式)
        """
        if defect_type_name not in self.defect_types:
            return []

        defect_type = self.defect_types[defect_type_name]
        crops = []

        for sample in defect_type.samples:
            if not sample.image_path.exists():
                continue

            # 画像を読み込み
            image_bgr = cv2.imread(str(sample.image_path))
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            crops.append(image_rgb)

        return crops

    def get_reference_samples_with_bbox(
        self, defect_type_name: str
    ) -> List[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        """
        カテゴリの全サンプルの画像とBBOX情報を取得

        Args:
            defect_type_name: 欠陥カテゴリ名

        Returns:
            List[Tuple[np.ndarray, Tuple]]: (画像, BBOX)のリスト
            画像はRGB形式、BBOXは(cx, cy, w, h)の正規化座標
        """
        if defect_type_name not in self.defect_types:
            return []

        defect_type = self.defect_types[defect_type_name]
        samples = []

        for sample in defect_type.samples:
            if not sample.image_path.exists():
                continue

            image_bgr = cv2.imread(str(sample.image_path))
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            samples.append((image_rgb, sample.bbox))

        return samples

    def list_defect_types(self) -> List[str]:
        """欠陥カテゴリ名のリストを取得"""
        return list(self.defect_types.keys())

    def get_sample_count(self, defect_type_name: str) -> int:
        """カテゴリのサンプル数を取得"""
        if defect_type_name not in self.defect_types:
            return 0
        return len(self.defect_types[defect_type_name].samples)

    def get_total_sample_count(self) -> int:
        """全サンプル数を取得"""
        return sum(len(dt.samples) for dt in self.defect_types.values())


def create_new_library(library_path: Path, library_name: str = "default") -> DefectLibrary:
    """
    新しい空のライブラリを作成

    Args:
        library_path: ライブラリの親ディレクトリ
        library_name: ライブラリ名

    Returns:
        DefectLibrary: 作成されたライブラリ
    """
    full_path = library_path / library_name
    full_path.mkdir(parents=True, exist_ok=True)
    return DefectLibrary(full_path)


def list_available_libraries(library_root: Path) -> List[str]:
    """
    利用可能なライブラリ一覧を取得

    Args:
        library_root: ライブラリのルートディレクトリ

    Returns:
        List[str]: ライブラリ名のリスト
    """
    if not library_root.exists():
        return []

    libraries = []
    for item in library_root.iterdir():
        if item.is_dir():
            # defect_config.jsonを持つサブディレクトリがあればライブラリとみなす
            has_defect_types = any(
                (item / subdir / "defect_config.json").exists()
                for subdir in item.iterdir()
                if subdir.is_dir()
            )
            if has_defect_types or not any(item.iterdir()):
                libraries.append(item.name)

    return sorted(libraries)
