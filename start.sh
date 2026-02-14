#!/bin/bash
# Oneshot AI Anomaly Detection 起動スクリプト
#
# 使用方法:
#   ./start.sh    # bashでコンテナに入る

# エラー発生時にスクリプトを即座に停止
set -e

# スクリプトのあるディレクトリに移動（どこから実行しても動作するように）
cd "$(dirname "$0")"

# X11転送を許可（コンテナからホストにGUIを表示するため）
xhost +local:docker > /dev/null 2>&1

# ホストユーザーのUID/GIDを取得（コンテナ内で同じ権限のユーザーを作成するため）
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)

# コンテナを起動してbashに入る（--rm: 終了時に自動削除）
docker compose run --rm anomaly-detection /bin/bash
