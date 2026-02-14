# Jetson フラッシュ & 環境構築ガイド

## Jetson フラッシュ手順

下記を参考に、デフォルトで入っている環境をフラッシュ。
https://wiki.seeedstudio.com/reComputer_Industrial_Getting_Started/


## 勝手にUpgradeされないように設定

ブート周りをアップグレードすると、Jetsonが起動しなくなってフラッシュする羽目になる。

### 1. Stop apt-daily timer

```bash
sudo systemctl disable --now apt-daily.timer
sudo systemctl disable --now apt-daily.service
sudo systemctl disable --now apt-daily-upgrade.timer
sudo systemctl disable --now apt-daily-upgrade.service
```

### 2. Hold Jetson packages

```bash
# ブートローダ・カーネル・デバイスツリーなど
sudo apt-mark hold 'nvidia-l4t-*'
```


## 検査アプリを動かすための環境構築

### 事前準備
パッケージリスト更新と基本ツールのインストール：

```bash
sudo apt update
sudo apt -y install curl gnupg lsb-release software-properties-common
```

### Firefox のインストール

```bash
sudo apt -y install firefox
firefox &
```

### Visual Studio Code のインストール
#### 1. **Microsoft GPG 鍵を追加**

```bash
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /usr/share/keyrings/microsoft.gpg > /dev/null
```

#### 2. **リポジトリ追加**（`arch=arm64` を明示）：

```bash
echo "deb [arch=arm64 signed-by=/usr/share/keyrings/microsoft.gpg]    https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
   ```

#### 3. **VS Code インストール**

```bash
sudo apt update
sudo apt -y install code
```

#### 4. **起動**

```bash
code &
```


### Githubとの連携

#### SSH key生成

```bash
ssh-keygen -t ed25519 -C "hayato.katsuma@smart-group.co.jp"
cat ~/.ssh/id_ed25519.pub
```

#### GithubへのSSH Keyの登録 

Account -> Settings -> SSH and GPG Keys -> New SSH Key


#### Username & emailのセッティング

```bash
git config --global user.name "HayatoKatsuma-SI"
git config --global user.email "hayato.katsuma@smart-group.co.jp"
```


#### アプリのコードをクローン

```bash
git clone git@github.com:si-bi-gryffindor/Kagome-Inspection-App.git
```


### Python 3.10 のインストール

```bash
sudo apt -y install python3.10 python3.10-venv python3.10-dev python3-pip python3-wheel

python3 --version   # → Python 3.10.x
pip3 --version
```

### その他パッケージ

```bash
sudo apt -y install git cmake build-essential  liblapack-dev libopenblas-dev libopenmpi-dev libomp-dev libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev swig python3-tk python3-pil.imagetk gstreamer1.0-tools gstreamer1.0-plugins-base ethtool
```

### NVIDIA JetPack

> **注意:** JetPack 6.x には CUDA 12.6 のみが含まれ、DL フレームワークは含まれません。
```bash
sudo apt -y install nvidia-jetpack
```

### Cuda Tool Kit

```bash
# JetPack 6.x は nvcc を明示的にインストールする必要がある
sudo apt update
sudo apt -y install cuda-toolkit-12-6     # JetPack 6.2 の場合

# シンボリックリンク & PATH / LD_LIBRARY_PATH
sudo ln -sf /usr/local/cuda-12.6 /usr/local/cuda
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### PyTorch 2.7.0 のインストール

#### 1. wheelファイルダウンロード

URLがアクセスできなくなったので、USB経由でwheelファイルをwheelフォルダに配置する。


#### 2. **JetPack 6.2（CUDA 12.6）用 wheel をインストール**

```bash
cd wheel
pip3 install torch-2.7.0-cp310-cp310-linux_aarch64.whl
pip3 install torchaudio-2.7.0-cp310-cp310-linux_aarch64.whl
pip3 install torchvision-0.22.0-cp310-cp310-linux_aarch64.whl
```

#### 3. **バージョン確認**

```bash
python3 - <<'PY'
import torch, torchvision
print("torch :", torch.__version__, "CUDA OK?", torch.cuda.is_available(), "cuDNN:", torch.backends.cudnn.version())
print("torchvision :", torchvision.__version__, "nms op:", hasattr(torch.ops.torchvision, "nms"))
PY
```

期待される出力：
```
torch : 2.7.0 CUDA OK? True cuDNN: 90300
torchvision : 0.22.0 nms op: True
```


### 追加 Python パッケージ

```bash
pip3 install --upgrade pip
cd ~/codes/Kagome-Inspection-App
pip3 install -r requirements.txt
```

### FAISS‑GPU のソースビルド

#### 1. **CMake 更新（推奨）**

##### Kitware リポジトリ追加

```bash
curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc   | gpg --dearmor -   | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"   | sudo tee /etc/apt/sources.list.d/kitware.list
```

##### 最新 CMakeをインストール
```bash
sudo apt update
sudo apt -y install kitware-archive-keyring
sudo apt -y install cmake
cmake --version
```

#### 2. **GPU 対応で FAISS をビルド**

```bash
# クローン
git clone https://github.com/facebookresearch/faiss.git

# ビルド
cmake -S faiss -B faiss/build -DFAISS_ENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES="87" -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF
cmake --build faiss/build -j$(nproc)

# Python バインディングインストール
cd faiss/build/faiss/python
python3 -m pip install .
```

### 最終環境確認

```bash
python3 - <<'PY'
import importlib, sys
libs = {
    'torch':       lambda m: f"{m.__version__} | CUDA={m.cuda.is_available()}",
    'torchvision': lambda m: m.__version__,
    'torchaudio':  lambda m: m.__version__,
    'faiss':       lambda m: f"{m.__version__} | GPU={'YES' if getattr(m,'get_num_gpus',lambda:0)()>0 else 'NO'}",
    'cv2':         lambda m: f"{m.__version__} | CUDA={'YES' if hasattr(m,'cuda') and getattr(m.cuda,'getCudaEnabledDeviceCount',lambda:0)()>0 else 'NO'}",
}
for name, info in libs.items():
    try:
        mod = importlib.import_module(name)
        print(f"{name:<11}: {info(mod)}")
    except ModuleNotFoundError:
        print(f"{name:<11}: NOT INSTALLED")
PY
```

期待される出力：

```
torch      : 2.7.0 | CUDA=True
torchvision: 0.22.0
torchaudio : 2.7.0
faiss      : 1.11.0 | GPU=YES
cv2        : 4.11.0 | CUDA=NO
```


---



## Baumer カメラ環境

### Baumer GAPI SDK（ARM64）のインストール

#### 1. **debファイルの取得**

すでに、codesファイルの中にBaumer_GAPI_SDK_2.15.2_lin_aarch64_cpp.debを格納済。
もし破損している場合は、下記のページからダウンロードしてください。
https://www.argocorp.com/software/DL/Baumer/software.html


#### 2. **インストール**

> **備考:** IPConfigTool も併せてインストールされます。
```bash
cd ~/codes
sudo apt install ./Baumer_GAPI_SDK_2.15.2_lin_aarch64_cpp.deb
```

### neoAPI のインストール

#### 1. **tarファイルの取得**

すでに、codesファイルの中にneoAPI_1.5.0_lin_aarch64_python.tar.gzを格納済。
もし破損している場合は、下記のページからダウンロードしてください。
https://www.argocorp.com/software/DL/Baumer/software.html

#### 2. **展開**

```bash
cd ~/codes
tar -xvzf neoAPI_1.5.0_lin_aarch64_python.tar.gz
```

#### 3. **neoAPI** インストール

```bash
cd ~/codes/Baumer_neoAPI_1.5.0_lin_aarch64_python/wheel
pip3 install baumer_neoapi-1.5.0-cp34.cp35.cp36.cp37.cp38.cp39.cp310.cp311.cp312-none-linux_aarch64.whl
```



### Camera Explorer のインストール

#### 1. **debファイルの取得**

すでに、codesファイルの中にCameraExplorer_3.5.2_lin_aarch64.debを格納済。
もし破損している場合は、下記のページからダウンロードしてください。
https://www.argocorp.com/software/DL/Baumer/software.html

#### 2. **インストール**
```bash
cd ~/codes
sudo apt install ./CameraExplorer_3.5.2_lin_aarch64.deb
```


### カメラ接続 & 給電

#### 1. **カメラの接続**

Jetson Orin NanoのLAN1にLANケーブルとカメラを接続する。

#### 2. **PoE給電開始**

##### ①gpioset を実行するシェルスクリプトを作成
```bash
sudo mkdir -p /usr/local/bin
sudo nano /usr/local/bin/enable_poe.sh
```

下記をenable_poe.shに追記し、Ctrl+X→Yで終了。
```bash
#!/usr/bin/env bash
# PoE 有効化 (GPIO chip 2, line 15 を High)
/usr/bin/gpioset 2 15=1
```

下記を実行。
```bash
sudo chmod +x /usr/local/bin/enable_poe.sh
```

##### ②systemd ユニットを作成
```bash
sudo nano /etc/systemd/system/enable-poe.service
```

下記をenable-poe.serviceに追記し、Ctrl+X→Yで終了。
```bash
[Unit]
Description=Enable PoE power for Baumer camera at boot
After=multi-user.target   # OS が起動しきったタイミングで実行

[Service]
Type=oneshot
ExecStart=/usr/local/bin/enable_poe.sh
RemainAfterExit=yes       # 実行後も「アクティブ」扱いにして再実行されないようにする

[Install]
WantedBy=multi-user.target
```

##### ③有効化と動作確認
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now enable-poe.service
systemctl status enable-poe.service      # Active (exited) になっていれば OK
sudo reboot                              # 再起動後も PoE ランプが点灯するか確認
```


### 固定 IP アドレスの割り当て

> カメラ情報IPv4 Address: 169.254.51.15  Subnet Mask: 255.255.0.0

#### 1. **カメラ NIC 確認**

```bash
ip -br link
```

#### 2. **接続プロファイル確認**（NIC `enP1p1s0` の例）

```bash
nmcli connection show | grep enP1p1s0
```

#### 3. **固定 IP を設定**（プロファイル名 `Wired connection 1 ` の例）

```bash
sudo nmcli connection modify 'Wired connection 1' ipv4.addresses 169.254.51.1/16 ipv4.method manual
sudo nmcli connection up 'Wired connection 1'
```

#### 4. **確認**（NIC `enP1p1s0` の例）

```bash
ip -4 addr show enP1p1s0
```

#### 5. **Camera Explorerによる確認**

Camera Explorerを起動して、カメラの映像が確認できれば、カメラが正常に接続できている。



### ジャンボフレームの有効化（永続設定）（プロファイル名 `Wired connection 1 ` の例）

```bash
nmcli connection show 'Wired connection 1'
sudo nmcli connection modify 'Wired connection 1' 802-3-ethernet.mtu 9000
sudo nmcli connection down 'Wired connection 1'
sudo nmcli connection up 'Wired connection 1'
nmcli connection show 'Wired connection 1' | grep -i mtu
# 9000に設定されてればOK。
```


### カメラ動作テスト

#### 1. **Camera Explorer** で **ExposureTime Auto** を **オフ** にしておく（オンのままだとエラー）。

#### 2. **テストスクリプト実行**

```bash
cd ~/codes/Baumer_neoAPI_1.5.0_lin_aarch64_python
python examples/getting_started.py
```

`getting_started.bmp` が生成されていればカメラ設定は **成功**。

---

