# Oneshot AI Anomaly Detection
# GPUノート (x86_64) + Ubuntu 22.04 + CUDA 12.8 環境向け

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8

# タイムゾーン設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 基本パッケージのインストール
# setup_manual_for_Jetson.mdの基本パッケージに対応
# Python 3.12をインストール（SAM3対応のため）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-tk \
    python3-pip \
    python3-wheel \
    git \
    cmake \
    sudo \
    gosu \
    ffmpeg \
    netcat \
    vim \
    wget \
    build-essential \
    liblapack-dev \
    libopenblas-dev \
    libopenmpi-dev \
    libomp-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    swig \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    ethtool \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    libxcb1 \
    pulseaudio \
    alsa-utils \
    locales \
    fonts-noto-cjk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 日本語ロケールの設定
RUN locale-gen ja_JP.UTF-8

# Python のデフォルトを python3.12 に設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# pip のインストール（Python 3.12用）
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    python3.12 -m pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.8 対応) のインストール
# nvidia/cuda:12.8.0ベースイメージに合わせてcu128を使用
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# CMakeを最新版に更新（faissビルドにCMake 3.24.0以上が必要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https ca-certificates gnupg && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y --no-install-recommends cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# FAISS-GPU のソースからビルド（Blackwell GPU対応）
# faiss-gpu-cu12はBlackwell (CC 12.0) 未対応のため、ソースからビルド
RUN git clone --depth 1 https://github.com/facebookresearch/faiss.git /tmp/faiss && \
    cd /tmp/faiss && \
    cmake -B build . \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="120" \
        -DPython_EXECUTABLE=$(which python3) && \
    make -C build -j$(nproc) faiss swigfaiss && \
    cd build/faiss/python && \
    pip3 install . && \
    rm -rf /tmp/faiss

# 作業ディレクトリの設定
WORKDIR /app

# requirements.txt をコピーしてインストール
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# SAM3関連の追加ライブラリをインストール（pyproject.toml準拠）
# 必須依存関係:
# - timm: PyTorch Image Models（メモリモジュール、DropPath等で使用）
# - huggingface_hub: モデルチェックポイントのダウンロード
# - iopath: ファイルパス管理
# - ftfy: テキスト正規化（BPEトークナイザーで使用）
# - regex: 正規表現（BPEトークナイザーで使用）
# - typing_extensions: 型ヒント拡張
# 追加依存関係:
# - einops: テンソル操作（RoPE実装で使用）
# - scikit-image: 連結成分解析（欠陥検出で使用）
# - setuptools: pkg_resources
# - decord: 動画読み込み（SAM3内部インポートで必要）
RUN pip3 install --no-cache-dir \
    timm>=1.0.17 \
    huggingface_hub>=0.20.0 \
    iopath>=0.1.10 \
    ftfy==6.1.1 \
    regex \
    typing_extensions \
    einops>=0.7.0 \
    scikit-image>=0.22.0 \
    setuptools>=69.0.0 \
    decord \
    pycocotools

# Baumer GAPI SDK (x86_64) のインストール
# libフォルダからコピーしてインストール
# Note: Dockerコンテナではudevが存在しないため、以下の対策が必要：
#   1. /etc/udev/rules.d/ ディレクトリを事前に作成
#   2. ダミーのudevadmコマンドを作成（post-installスクリプトが呼び出すため）
COPY lib/Baumer_GAPI_SDK_2.15.2_lin_x86_64_cpp.deb /tmp/
RUN mkdir -p /etc/udev/rules.d && \
    printf '#!/bin/bash\nexit 0\n' > /usr/bin/udevadm && \
    chmod +x /usr/bin/udevadm && \
    apt-get update && \
    apt-get install -y /tmp/Baumer_GAPI_SDK_2.15.2_lin_x86_64_cpp.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm /tmp/Baumer_GAPI_SDK_2.15.2_lin_x86_64_cpp.deb

# neoAPI (Baumer カメラ Python SDK) のインストール
# libフォルダからコピーしてインストール
COPY lib/baumer_neoapi-1.5.0-cp34.cp35.cp36.cp37.cp38.cp39.cp310.cp311.cp312-none-linux_x86_64.whl /tmp/
RUN pip3 install /tmp/baumer_neoapi-1.5.0-cp34.cp35.cp36.cp37.cp38.cp39.cp310.cp311.cp312-none-linux_x86_64.whl && \
    rm /tmp/baumer_neoapi-1.5.0-cp34.cp35.cp36.cp37.cp38.cp39.cp310.cp311.cp312-none-linux_x86_64.whl

# DINOv2モデルのキャッシュディレクトリはentrypoint.shで動的に設定
# （/tmp/torch_cacheに設定して権限エラーを回避）

# entrypoint.sh をコピー
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# デフォルトコマンド
CMD ["python", "main.py"]
