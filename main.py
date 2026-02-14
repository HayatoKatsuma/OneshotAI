# main.py ───────────────────────────────────────────────────────────────
"""
Baumer 産業カメラ × AnomalyDINO / SAM3
検査アプリケーション（TkEasyGUI 版）

────────────────────────────────────────────
■ 本ファイルの構成
    1. ModeSelectApp       : モード選択画面
    2. main()              : エントリポイント

■ 各モードのアプリ
    Anomaly_Detection/app_anomaly_inspection.py : 異常検知（フィルム取り違え検査）
    Segmentation/app_defect_inspection.py       : 欠陥検出・計数
────────────────────────────────────────────

"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import TkEasyGUI as eg


# =============================================================================
# ModeSelectApp クラス（モード選択画面）
# =============================================================================
class ModeSelectApp:
    """起動時のモード選択画面"""

    def __init__(self, root: Path):
        self.root = root
        self.sys = self._load_json(root / "config" / "system_params.json")
        self.win_size = tuple(self.sys.get("window_size", [1600, 900]))

        try:
            eg.set_theme("clam")
        except Exception:
            pass

    def _load_json(self, path: Path) -> Dict:
        """設定ファイル読み込み"""
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _make_window(self) -> eg.Window:
        """モード選択画面を生成"""
        layout = [
            [eg.Text("Oneshot AI", font=("Arial", 32))],
            [eg.HSeparator()],
            [eg.Text("モード選択", font=("Arial", 24))],
            [eg.Text("")],
            [
                eg.Button(
                    "異常検知モード",
                    key="-ANOMALY-",
                    font=("Arial", 16),
                    size=(25, 4),
                ),
            ],
            [eg.Text("")],
            [
                eg.Button(
                    "欠陥検出・計数モード",
                    key="-DEFECT-",
                    font=("Arial", 16),
                    size=(25, 4),
                ),
            ],
            [eg.Text("")],
            [eg.HSeparator()],
            [eg.Button("終了", key="-EXIT-", font=("Arial", 12))],
        ]

        win = eg.Window(
            "検査アプリケーション",
            layout,
            finalize=True,
            size=(500, 700),
            element_justification="center",
        )
        return win

    def run(self) -> str | None:
        """
        モード選択を実行

        Returns:
            str | None: 選択されたモード ("anomaly" / "defect")、終了時はNone
        """
        win = self._make_window()

        while True:
            ev, _ = win.read()

            if ev in (eg.WINDOW_CLOSED, "-EXIT-"):
                win.close()
                return None

            if ev == "-ANOMALY-":
                win.close()
                return "anomaly"

            if ev == "-DEFECT-":
                win.close()
                return "defect"


# =============================================================================
# エントリポイント
# =============================================================================
def main():
    """
    アプリケーションのエントリポイント

    1. モード選択画面を表示
    2. 選択されたモードに応じたアプリを起動
    """
    root = Path(__file__).parent

    while True:
        # モード選択
        mode_selector = ModeSelectApp(root)
        mode = mode_selector.run()

        if mode is None:
            # 終了が選択された
            break

        if mode == "anomaly":
            # 異常検知モード
            from Anomaly_Detection.app_anomaly_inspection import AnomalyInspectionApp
            AnomalyInspectionApp(root).run()

        elif mode == "defect":
            # 欠陥検出・計数モード
            from Segmentation.app_defect_inspection import DefectInspectionApp
            DefectInspectionApp(root).run()

        # モード終了後、再度モード選択画面に戻る


if __name__ == "__main__":
    main()
