# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

フィルム取り違え検査システム。AnomalyDINO（DINOv2ベースの異常検出）とBaumer産業カメラを組み合わせ、撮影画像とマスター画像のk-NN類似度スコアでOK/NGを判定する。

## Commands

### Docker環境での実行（推奨）
```bash
# コンテナ起動（bashでログイン）
./start.sh

# コンテナ内で実行
python main.py              # メイン検査GUI
python app_capture.py       # 画像撮影ユーティリティ
python app_generate_master.py    # マスター特徴量生成GUI
```

### ネイティブ環境での実行
```bash
pip install -r requirements.txt
# Note: neoapi (Baumer camera SDK) は別途インストールが必要

python main.py
python app_capture.py
```

### 分析ツール
```bash
# Anomaly_Detection内で実行
cd Anomaly_Detection
python analyze_score.py --target_dir ../data/2025_07_17 --masters A --roi_rel 0.70 0.93 0.11 0.39
```

## Architecture

### ディレクトリ構成
```
├── main.py                    # メイン検査アプリ（TkEasyGUI）
├── app_capture.py             # 画像撮影ユーティリティ
├── app_generate_master.py     # マスター特徴量生成GUI
├── utils/
│   ├── baumer_camera.py       # BaumerCamera: neoAPIラッパー
│   └── anomaly_dino_detector.py  # 再エクスポート用スタブ
├── Anomaly_Detection/
│   ├── src/backbones.py       # DINOv2/ViTモデルラッパー
│   ├── utils/anomaly_dino_detector.py  # k-NN異常検出クラス
│   ├── analyze_score.py       # バッチ分析ツール
│   └── master/<product>/      # 製品別マスター（master.bmp, master.npy, master_hue.txt）
├── Segmentation/              # SAM3セグメンテーション（実験用）
├── config/                    # JSON設定ファイル
├── logs/                      # NG検出ログ
└── docker/                    # Docker環境設定
```

### 検出パイプライン
1. `BaumerCamera.grab_nonblock()` → RGB画像取得
2. `DINOv2Wrapper.extract_features()` → 中間層特徴抽出（`feat_layer`で指定）
3. `AnomalyDINODetector._score_single()` → FAISSによるk-NN距離計算
4. 最終スコア = `min_dino_score + hue_weight * hue_score`

### 設定ファイル
- `config/base_camera_params.json` - Baumerカメラのデフォルト設定
- `config/base_model_params.json` - モデルパラメータのデフォルト設定
- `config/system_params.json` - GUIサイズ、バッファ設定、ログ上限
- `Anomaly_Detection/master/<product>/camera_params.json` - 製品別カメラ設定（上書き）
- `Anomaly_Detection/master/<product>/model_params.json` - 製品別threshold設定（上書き）

### 主要パラメータ

| パラメータ | 説明 |
|-----------|------|
| `model_type` | DINOv2バリアント（dinov2_vits14等） |
| `feat_layer` | 特徴抽出層（0〜11） |
| `image_size` | 入力画像サイズ（patch_size=14の倍数） |
| `threshold` | OK/NG判定閾値 |
| `roi_rel` | Hue計算用ROI（y_start, y_end, x_start, x_end の相対座標） |
| `sat_thresh` | Hue計算対象の最小彩度 |
| `hue_weight` | Hueスコアの重み係数 |

## Docker環境

```bash
# ビルド
docker compose build

# 起動（GUI付き）
./start.sh

# 手動起動
xhost +local:docker
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm anomaly-detection
```

- CUDA 12.8 + Python 3.12
- GPU: NVIDIA Container Toolkit必須
- GUI: X11ソケットをマウント
- カメラ: /devをprivilegedモードでマウント

## Notes

- 画像は内部的にRGB形式で処理（BaumerCameraがBGR→RGB変換を担当）
- マスター特徴量(.npy)はFAISS用にL2正規化済み
- NG検出時はBuzzer.wavを`paplay`で再生（Linux環境）

## 指示

- 常に日本語で応答してください
- コードのコメントも日本語で書いてください
- エラーメッセージの説明も日本語でお願いします
