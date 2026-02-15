# Oneshot AI 検査システム

Baumer産業カメラとDINOv2ベースのAI検出を組み合わせた、ワンショット検査システム。
2つの検査モード（異常検知・欠陥検出）を備え、マスター登録から検査までの一連のワークフローをGUIで提供する。

## システム構成

```
main.py（モード選択画面）
  ├── 異常検知モード ─── Anomaly_Detection/app_anomaly_inspection.py
  └── 欠陥検出モード ─── Segmentation/app_defect_inspection.py

準備ツール
  ├── 異常検知用マスター管理 ─── Anomaly_Detection/app_anomaly_master.py
  └── 欠陥検出用サンプル登録 ─── Segmentation/app_defect_registration.py
```

## 実行環境

### Docker環境（推奨）

```bash
# コンテナ起動
./start.sh

# コンテナ内で各アプリを実行
python main.py
```

### 必要要件

- CUDA 12.8 + Python 3.12
- NVIDIA Container Toolkit（GPU使用時）
- X11ソケット（GUI表示用）
- Baumer neoAPI（カメラ接続時）

---

## 異常検知モード

マスター画像と検査画像のDINOv2特徴量をk-NN比較し、類似度スコアでOK/NGを判定する。
フィルムの取り違え検査など、マスターとの一致度を確認する用途に使用する。

### 判定ロジック

```
最終スコア = DINOスコア（k-NN距離） + hue_weight × Hueスコア（色相差）
判定: スコア < threshold → OK / スコア >= threshold → NG
```

### 1. マスター登録（app_anomaly_master.py）

検査に使用するマスター画像の撮影と、特徴量ファイルの生成を行う。

```bash
python Anomaly_Detection/app_anomaly_master.py
```

#### 操作手順

1. **製品を選択または新規作成**
   - 起動すると製品選択画面が表示される
   - 既存の製品を選択するか、「新規作成」で製品名を入力して作成する

2. **マスター画像を撮影**
   - メイン画面の「撮影」ボタンでカメラから1枚撮影
   - 撮影した画像はプレビュー表示され、`master_YYYYMMDD_HHMMSS.bmp` として保存される
   - 必要な枚数分、繰り返し撮影する（複数枚登録で検出精度が向上する）
   - 不要な画像は一覧から選択して削除できる

3. **特徴量を生成**
   - 「特徴量生成」ボタンを押すと以下のファイルが自動生成される
     - `master.npy` — DINOv2特徴量（FAISS用、L2正規化済み）
     - `master_hue.txt` — ROI領域の平均Hue値（色相判定用）
   - プログレスバーで進捗を確認できる

4. **完了後、「終了」で製品選択に戻る**

#### 生成されるファイル

```
Anomaly_Detection/master/<製品名>/
  ├── master_20260215_143022.bmp   # マスター画像（複数可）
  ├── master.npy                    # DINOv2特徴量
  └── master_hue.txt                # 平均Hue値
```

### 2. 検査実行（main.py → 異常検知モード）

```bash
python main.py
# → 「異常検知モード」を選択
```

#### 操作手順

1. **品種選択画面**で検査対象の製品を選択
2. **検査画面**が開き、マスター画像が左側に表示される
3. **手動モード**: 「撮像＆判定」ボタンで1枚撮影→判定
4. **トリガーモード**: 外部トリガー信号で自動撮影→判定
5. NG判定時はブザー音が鳴り、ヒートマップ付き画像がログに保存される
6. 検査画面でしきい値を直接変更できる（入力欄に値を入力→「適用」）

#### 設定ファイル

| ファイル | 用途 |
|---------|------|
| `config/base_camera_params.json` | カメラ共通設定 |
| `Anomaly_Detection/config/base_model_params.json` | モデル・しきい値の共通設定 |
| `Anomaly_Detection/master/<製品>/camera_params.json` | 製品別カメラ設定（上書き） |
| `Anomaly_Detection/master/<製品>/model_params.json` | 製品別しきい値設定（上書き） |
| `config/system_params.json` | GUI表示サイズ、バッファ、ログ上限 |

#### 主要パラメータ

| パラメータ | 説明 |
|-----------|------|
| `model_type` | DINOv2バリアント（`dinov2_vits14` 等） |
| `feat_layer` | 特徴抽出層（0〜11） |
| `image_size` | 入力画像サイズ（patch_size=14の倍数） |
| `threshold` | OK/NG判定しきい値 |
| `roi_rel` | Hue計算用ROI（`[y_start, y_end, x_start, x_end]` の相対座標） |
| `sat_thresh` | Hue計算対象の最小彩度 |
| `hue_weight` | Hueスコアの重み係数 |

---

## 欠陥検出モード

SAM3（Segment Anything Model 3）のVisual Prompt機能を使い、リファレンス画像の欠陥パターンと同じ欠陥を検査画像から検出・計数する。
傷、穴、汚れなど、複数種類の欠陥を同時に検出できる。

### 判定ロジック

```
各欠陥タイプごとに:
  SAM3がリファレンスBBOXを元に検査画像内の類似領域を検出
  → confidence_score >= threshold のものを欠陥として計上
判定: 検出数 > 0 → NG / 検出数 = 0 → OK
```

### 1. 欠陥サンプル登録（app_defect_registration.py）

検出対象とする欠陥のリファレンス画像とBBOX（バウンディングボックス）を登録する。

```bash
python Segmentation/app_defect_registration.py
```

#### 操作手順

1. **ライブラリを選択または新規作成**
   - 起動するとライブラリ選択画面が表示される
   - 既存のライブラリを選択するか、「新規作成」で作成する
   - ライブラリは検査対象（製品）ごとに作成する

2. **欠陥タイプを作成**
   - 右側パネルの「新規タイプ」ボタンで欠陥カテゴリを追加
   - プリセット（傷、穴、汚れ、ひび、へこみ等）から選択するか、カスタム名を入力
   - 確信度しきい値（0〜1）を設定する

3. **サンプル画像を読み込む**
   - 「画像読込」でファイルから画像を読み込む
   - 「撮影」でカメラから直接撮影する

4. **BBOXを描画する**
   - 画像上で検出したい欠陥領域をドラッグして矩形選択
   - 赤枠でプレビューされ、確定すると緑枠に変わる
   - 「BBOX解除」で選択をやり直せる

5. **サンプルを登録する**
   - 右側パネルから欠陥タイプを選択
   - 「サンプル登録」ボタンで画像とBBOXがライブラリに保存される

#### 生成されるファイル

```
Segmentation/defect_library/<ライブラリ名>/
  └── <欠陥タイプ名>/
      ├── defect_config.json        # タイプ設定（しきい値、色等）
      └── samples/
          └── <sample_id>.png       # リファレンス画像
```

### 2. 検査実行（main.py → 欠陥検出モード）

```bash
python main.py
# → 「欠陥検出・計数モード」を選択
```

#### 操作手順

1. **ライブラリ選択画面**で使用する欠陥ライブラリを選択
2. **検査画面**が開き、登録済みの欠陥クラス一覧が右側に表示される
3. **手動モード**: 「撮像＆検出」ボタンで1枚撮影→検出、または「画像読込」でファイルから検査
4. **トリガーモード**: 外部トリガー信号で自動撮影→検出
5. 検出結果は判定（OK/NG）、欠陥総数、クラス別内訳として表示される
6. NG判定時はブザー音が鳴り、結果画像とCSVログが保存される
7. 検査画面で各欠陥タイプのしきい値を直接変更できる（入力欄に値を入力→「閾値を適用」）

#### 設定ファイル

| ファイル | 用途 |
|---------|------|
| `config/base_camera_params.json` | カメラ共通設定 |
| `Segmentation/config/defect_params.json` | 欠陥検出パラメータ（表示サイズ、色設定等） |

---

## 分析ツール

### analyze_score.py

指定ディレクトリ内のBMP画像に対して異常度計算を行い、時系列プロットを生成するCLIツール。

```bash
cd Anomaly_Detection
python analyze_score.py \
    --target_dir ../data/2025_07_17 \
    --masters A B \
    --roi_rel 0.70 0.93 0.11 0.39 \
    --sat_thresh 40 \
    --hue_weight 0.10
```

#### 出力ファイル

- 時系列プロット画像（PNG形式）
- 最小異常度画像（ROI表示付き、BMP形式）
- `analyze/` フォルダに実行日時付きで保存

---

## ディレクトリ構成

```
Oneshot_AI/
├── main.py                              # モード選択画面（エントリポイント）
├── app_capture.py                       # 画像撮影ユーティリティ
├── config/
│   ├── base_camera_params.json          # カメラ共通設定
│   └── system_params.json               # GUI・システム設定
├── utils/
│   └── baumer_camera.py                 # BaumerCamera: neoAPIラッパー
├── Anomaly_Detection/
│   ├── app_anomaly_inspection.py        # 異常検知 検査アプリ
│   ├── app_anomaly_master.py            # マスター画像管理アプリ
│   ├── app_generate_master.py           # マスター特徴量一括生成（旧ツール）
│   ├── analyze_score.py                 # バッチ分析ツール
│   ├── config/
│   │   └── base_model_params.json       # モデル共通設定
│   ├── master/<製品名>/                  # 製品別マスターデータ
│   ├── src/backbones.py                 # DINOv2モデルラッパー
│   ├── utils/anomaly_dino_detector.py   # k-NN異常検出クラス
│   └── logs/                            # NG検出ログ
├── Segmentation/
│   ├── app_defect_inspection.py         # 欠陥検出 検査アプリ
│   ├── app_defect_registration.py       # 欠陥サンプル登録アプリ
│   ├── config/
│   │   └── defect_params.json           # 欠陥検出パラメータ
│   ├── defect_library/<ライブラリ名>/    # 欠陥ライブラリ
│   ├── utils/
│   │   ├── defect_detector.py           # SAM3欠陥検出クラス
│   │   └── defect_library.py            # ライブラリ管理クラス
│   └── logs/                            # 検査結果CSV
└── docker/                              # Docker環境設定
```
