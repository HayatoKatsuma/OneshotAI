# RTX 5080 Laptop 環境構築ガイド（Docker版）

このマニュアルは、ASUS ROG Strix G16 (G615LW) に Ubuntu 22.04 LTS をインストールし、NVIDIA Docker や VS Code 等の開発環境を構築し、Oneshot AI Anomaly 検査アプリをDockerコンテナ上で動作させるための手順書です。

---

## 1. Ubuntu 22.04 ISOファイルの取得

1. [Ubuntu 22.04.x LTS (Jammy Jellyfish) Releases](https://releases.ubuntu.com/22.04/) にアクセス。
2. **Desktop image** (`ubuntu-22.04.x-desktop-amd64.iso`) をダウンロード。

---

## 2. インストールメディアの作成 (Windows)

1. [Rufus公式サイト](https://rufus.ie/ja/)から最新版をダウンロード。
2. **デバイス:** 8GB以上のUSBメモリを選択。
3. **ブートの種類:** ダウンロードしたISOを選択。
4. **パーティション構成:** **GPT**。
5. **ターゲットシステム:** **UEFI (CSM無効)**。
6. **書き込み:** 「スタート」をクリックし、「ISOイメージモード」で実行。

---

## 3. BIOS (UEFI) 設定

ASUS特有の制限を解除します。

1. 電源ON直後に **[F2]** を連打してBIOS画面へ。
2. **Advanced Mode (F7)** を開く。
3. **Advanced > VMD Setup Menu:** `VMD Controller` を **Disabled** に設定（SSD認識のため）。
4. **Security > Secure Boot:** **Disabled** に設定。
5. **Boot > Fast Boot:** **Disabled** に設定。
6. **[F10]** で保存して終了。

---

## 4. Ubuntuのインストール

1. **[Esc]** を連打してブートメニューからUSBを選択。
2. `Try or Install Ubuntu` を選択。
3. インストーラーで以下の点に注意：
   - **通常のインストール** を選択。
   - **「グラフィックスとWi-Fiハードウェア...のサードパーティ製ソフトウェアをインストールする」** に必ずチェック。
   - **インストールの種類:** 「ディスクを削除してUbuntuをインストール」を選択。

---

## 5. ログインループの解決（初回起動時）

パスワード入力後に画面が戻ってしまう場合の対処です。

1. ログイン画面で **[Ctrl] + [Alt] + [F3]** を押し、CUIにログイン。
2. 以下のコマンドでNVIDIAドライバを導入：
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-570-open
   sudo reboot
   ```

---

## 6. 電源終了時のフリーズ対策

1. 端末（Terminal）を開き、GRUBを編集：
   ```bash
   sudo nano /etc/default/grub
   ```

2. `GRUB_CMDLINE_LINUX_DEFAULT` を書き換え：
   ```
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash acpi=force reboot=pci"
   ```

3. 反映とサービスの有効化：
   ```bash
   sudo update-grub
   sudo systemctl enable nvidia-suspend.service nvidia-hibernate.service nvidia-resume.service
   ```

---

## 7. Google Chrome のインストール

```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
rm google-chrome-stable_current_amd64.deb
```

---

## 8. Git の設定

```bash
sudo apt install git -y
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"

# SSH鍵作成
ssh-keygen -t ed25519 -C "your-email@example.com"
# GitHubに登録するための公開鍵を表示
cat ~/.ssh/id_ed25519.pub
```

GitHubの **Account > Settings > SSH and GPG Keys > New SSH Key** で公開鍵を登録。

---

## 9. Visual Studio Code のインストール

```bash
sudo apt update && sudo apt install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update && sudo apt install code
```

---

## 10. NVIDIA Container Toolkit (NVIDIA Docker)

### 10.1 Docker Engine

```bash
sudo apt install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update && sudo apt install docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER
newgrp docker
```

### 10.2 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 10.3 動作確認

```bash
# Docker内でGPUが認識されるかテスト
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 11. リポジトリのクローン

```bash
mkdir -p ~/codes
cd ~/codes
git clone git@github.com:si-bi-gryffindor/Oneshot_AI_Anomaly.git
```

---

## 12. Baumer カメラ SDK の準備

### 12.1 プロジェクト内のSDKファイル

プロジェクトの `lib/` フォルダに以下のSDKファイルが格納されています：

```
lib/
├── Baumer_GAPI_SDK_2.15.2_lin_x86_64_cpp.deb    # Baumer GAPI SDK (x86_64)
├── baumer_neoapi-1.5.0-...-linux_x86_64.whl     # neoAPI Python SDK
└── CameraExplorer_3.5.2_lin_x86_64.deb          # Camera Explorer
```

> **注意**: ファイルが不足している場合は、Baumer公式サイトまたはArgo社からダウンロードしてください。
> ダウンロードURL: https://www.argocorp.com/software/DL/Baumer/software.html

### 12.2 ホストOSへの SDK インストール（カメラ動作確認用）

Camera Explorerでカメラの動作確認を行うため、ホストOSにSDKをインストールします：

```bash
cd ~/codes/Oneshot_AI_Anomaly

# Baumer GAPI SDK のインストール
sudo apt install ./lib/Baumer_GAPI_SDK_2.15.2_lin_x86_64_cpp.deb

# Camera Explorer のインストール
sudo apt install ./lib/CameraExplorer_3.5.2_lin_x86_64.deb
```

> **備考**: neoAPI（Pythonライブラリ）はDockerイメージのビルド時に自動的にインストールされるため、ホストOSへのインストールは不要です。

---

## 13. カメラ接続設定

### 13.1 カメラ接続

LANケーブルでカメラをPCに接続。PoE対応の場合はPoEインジェクタを使用。

### 13.2 NIC 確認

```bash
ip -br link
```

カメラ接続用NIC名を確認（例: `enp130s0`）。

### 13.3 接続プロファイル確認

```bash
nmcli connection show | grep <NIC名>
```

### 13.4 固定 IP アドレス設定

> カメラ情報: IPv4 Address: 169.254.51.15, Subnet Mask: 255.255.0.0

```bash
# 固定IP設定（プロファイル名は環境に応じて変更）
sudo nmcli connection modify 'Wired connection 1' ipv4.addresses 169.254.51.1/16 ipv4.method manual
sudo nmcli connection up 'Wired connection 1'

# 確認
ip -4 addr show enp130s0
```

### 13.5 ジャンボフレームの有効化

```bash
sudo nmcli connection modify 'Wired connection 1' 802-3-ethernet.mtu 9000
sudo nmcli connection down 'Wired connection 1'
sudo nmcli connection up 'Wired connection 1'
nmcli connection show 'Wired connection 1' | grep -i mtu
# 9000に設定されていればOK
```

### 13.6 Camera Explorer での動作確認

カメラ映像が表示されれば接続成功。**ExposureTime Auto** を **Off** に設定しておく。

---

## 14. Docker イメージのビルド

### 14.1 Docker Compose を使用したビルド（推奨）

```bash
cd ~/codes/Oneshot_AI_Anomaly
docker compose build
```

### 14.2 手動ビルド（オプション）

```bash
cd ~/codes/Oneshot_AI_Anomaly
docker build -f docker/Dockerfile -t oneshot-ai-anomaly:latest .
```

### 14.3 ビルド確認

```bash
docker images | grep oneshot-ai-anomaly
```

> **備考**: ビルド時に `lib/` フォルダ内のBaumer SDK と neoAPI が自動的にインストールされます。

---

## 15. Docker コンテナの起動

### 15.1 起動スクリプトを使用（推奨）

```bash
cd ~/codes/Oneshot_AI_Anomaly

# アプリを起動
./start.sh

# bashでコンテナに入る場合
./start.sh bash
```

> **備考**:
> - 起動スクリプトは X11転送許可、環境変数設定を自動で行います。
> - Ctrl+C で停止すると自動的にコンテナが削除されます。

### 15.2 手動で起動する場合

```bash
cd ~/codes/Oneshot_AI_Anomaly

# X11転送を許可（GUIアプリ表示に必要）
xhost +local:docker

# アプリを起動
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm anomaly-detection

# bashでコンテナに入る場合
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm anomaly-detection /bin/bash
```

### 15.3 docker composeを使わない場合（オプション）

```bash
xhost +local:docker
docker run -it --rm \
    --gpus all \
    --net=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e LOCAL_UID=$(id -u) \
    -e LOCAL_GID=$(id -g) \
    -e LOCAL_USER=$USER \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/app \
    -v /dev:/dev \
    oneshot-ai-anomaly:latest \
    python main.py
```

**オプション説明:**
- `--rm`: コンテナ終了時に自動削除
- `--gpus all`: GPUアクセス
- `--net=host`: ホストネットワーク共有（カメラ通信用）
- `--privileged`: デバイスアクセス権限
- `-v /dev:/dev`: カメラデバイスマウント
- `-e DISPLAY`: X11ディスプレイ転送
- `-e LOCAL_UID/LOCAL_GID`: ファイル権限をホストユーザーに合わせる

---

## 16. 環境確認スクリプト

コンテナ内で以下を実行して環境を確認：

```bash
python3 - <<'PY'
import importlib
libs = {
    'torch':       lambda m: f"{m.__version__} | CUDA={m.cuda.is_available()}",
    'torchvision': lambda m: m.__version__,
    'torchaudio':  lambda m: m.__version__,
    'faiss':       lambda m: f"{m.__version__} | GPU={'YES' if getattr(m,'get_num_gpus',lambda:0)()>0 else 'NO'}",
    'cv2':         lambda m: m.__version__,
    'neoapi':      lambda m: "OK",
}
for name, info in libs.items():
    try:
        mod = importlib.import_module(name)
        print(f"{name:<11}: {info(mod)}")
    except ModuleNotFoundError:
        print(f"{name:<11}: NOT INSTALLED")
    except Exception as e:
        print(f"{name:<11}: ERROR - {e}")
PY
```

期待される出力（CUDA 12.8 環境）：
```
torch      : 2.7.x | CUDA=True
torchvision: 0.22.x
torchaudio : 2.7.x
faiss      : 1.x.x | GPU=YES
cv2        : 4.x.x
neoapi     : OK
```

---

## 17. トラブルシューティング

### GUI が表示されない

```bash
# X11転送を再許可
xhost +local:docker

# DISPLAYが正しく設定されているか確認
echo $DISPLAY
```

### カメラが認識されない

1. ホストOSでCamera Explorerが動作するか確認
2. `--privileged` と `--net=host` オプションが付いているか確認
3. 固定IPとジャンボフレーム設定を再確認

### GPU が認識されない

```bash
# ホストでGPU確認
nvidia-smi

# Docker + NVIDIA Toolkitの動作確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 18. 参考リンク

- [Baumer ソフトウェアダウンロード](https://www.argocorp.com/software/DL/Baumer/software.html)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
