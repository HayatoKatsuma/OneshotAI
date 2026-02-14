#!/bin/bash

# ホストのユーザーID/グループIDでコンテナ内にユーザーを作成
# これによりマウントしたファイルの権限問題を回避

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}
USER_NAME=${LOCAL_USER:-user}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID, USER: $USER_NAME"

# ユーザーが既に存在する場合はスキップ
if ! id "$USER_NAME" &>/dev/null; then
    useradd -u $USER_ID -o -m $USER_NAME
    groupmod -g $GROUP_ID $USER_NAME
    echo "$USER_NAME:$USER_NAME" | chpasswd
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

# オーディオグループに追加（ブザー音再生用）
adduser $USER_NAME audio 2>/dev/null || true

export HOME=/home/$USER_NAME
export SHELL=/bin/bash

# PyTorchのキャッシュディレクトリを/tmpに設定（権限エラー回避）
export TORCH_HOME=/tmp/torch_cache
mkdir -p $TORCH_HOME
chown -R $USER_ID:$GROUP_ID $TORCH_HOME

# 指定されたコマンドをユーザー権限で実行
exec /usr/sbin/gosu $USER_NAME "$@"
