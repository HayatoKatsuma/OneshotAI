# UGREEN AC650 (RTL8811CU) Driver — Jetson Orin Nano / JetPack 6.2 最速インストールガイド

> **対象環境**  
> • Jetson Linux R36.4.3 (JetPack 6.2)  
> • Ubuntu 22.04 (aarch64)  
> • UGREEN AC650 USB Wi‑Fi（Realtek RTL8811CU）

---

## 0. 依存パッケージ一括導入

```bash
sudo apt update
sudo apt install -y build-essential bc flex bison libssl-dev libelf-dev dwarves git wget
```

---

## 1. カーネルソースを取得 & 展開

```bash
cd /usr/src
sudo wget -O public_sources.tbz2 \
  https://developer.download.nvidia.com/embedded/L4T/r36_Release_v4.3/sources/public_sources.tbz2

# kernel_src.tbz2 を抽出 → 展開
sudo tar xf public_sources.tbz2 \
  Linux_for_Tegra/source/kernel_src.tbz2 --strip-components=2
sudo tar xf kernel_src.tbz2                # → /usr/src/kernel/

# ソースルートを環境変数に
export SRC_DIR=/usr/src/kernel/kernel-jammy-src
```

---

## 2. カーネルビルド環境を準備

```bash
zcat /proc/config.gz | sudo tee $SRC_DIR/.config >/dev/null
sudo make -C $SRC_DIR O=$SRC_DIR -j$(nproc) olddefconfig
sudo make -C $SRC_DIR O=$SRC_DIR -j$(nproc) prepare scripts modules_prepare
```

---

## 3. 8821cu ドライバをビルド

```bash
git clone https://github.com/morrownr/8821cu-20210916.git ~/8821cu
cd ~/8821cu && make clean

make -j$(nproc) ARCH=arm64 \
     -C $SRC_DIR M=$(pwd) modules
```

---

## 4. ドライバ導入 & 反映

```bash
sudo cp 8821cu.ko /lib/modules/$(uname -r)/kernel/drivers/net/wireless/
sudo depmod -a
sudo modprobe 8821cu        # 今回だけ手動ロード
```

> 次回ブート以降は自動ロードされます。

---

## 5. Wi‑Fi 接続例

```bash
# インタフェース名を自動取得
IF=$(ls /sys/class/net | grep -E '^wlx|^wlp' | head -n1)

# AP 一覧を確認
nmcli device wifi list ifname $IF

# 接続
nmcli device wifi connect "<SSID>" password "<PASSWORD>" ifname $IF
```

---

## 6. よくある追加設定（任意）

| 目的            | コマンド |
|-----------------|----------|
| 省電力 OFF      | `sudo iw dev $IF set power_save off` |
| 自動接続有効    | `nmcli connection modify "<SSID>" connection.autoconnect yes` |

---

## 7. カーネル更新時（JetPack 小アップデート）

1. 新しい **public_sources.tbz2** を再取得・展開  
2. **手順 2** を実行（`prepare scripts modules_prepare`）  
3. **手順 3–4** で `.ko` を再ビルド → コピー  

> **備考**  
> * Flash し直しは不要  
> * 省電力設定は AP プロファイルごとに保持  
> * `depmod -a` 済みなので再起動後も自動ロード
