#!/bin/bash

echo "Kagome検査アプリのデスクトップアイコン設定を開始します..."

# 現在のディレクトリを確認
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_FILE="$SCRIPT_DIR/Kagome-Inspection.desktop"
DESKTOP_DIR="$HOME/Desktop"
APPLICATIONS_DIR="$HOME/.local/share/applications"

# デスクトップディレクトリが存在しない場合は作成
if [ ! -d "$DESKTOP_DIR" ]; then
    echo "デスクトップディレクトリを作成しています..."
    mkdir -p "$DESKTOP_DIR"
fi

# アプリケーションディレクトリが存在しない場合は作成
if [ ! -d "$APPLICATIONS_DIR" ]; then
    echo "アプリケーションディレクトリを作成しています..."
    mkdir -p "$APPLICATIONS_DIR"
fi

# デスクトップにアイコンをコピー
echo "デスクトップにアイコンをコピーしています..."
cp "$DESKTOP_FILE" "$DESKTOP_DIR/"
chmod +x "$DESKTOP_DIR/Kagome-Inspection.desktop"

# アプリケーションメニューにも追加
echo "アプリケーションメニューに追加しています..."
cp "$DESKTOP_FILE" "$APPLICATIONS_DIR/"
chmod +x "$APPLICATIONS_DIR/Kagome-Inspection.desktop"

echo "設定完了！"
echo ""
echo "デスクトップにKagome検査アプリのアイコンが作成されました。"
echo "アイコンをダブルクリックするとアプリが起動します。"
echo ""
echo "もしアイコンが表示されない場合は、以下を試してください："
echo "1. デスクトップで右クリック → リフレッシュ"
echo "2. ファイルマネージャーでデスクトップフォルダを開く"