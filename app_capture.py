#!/usr/bin/env python3
"""
TkEasyGUIを使った画像撮影アプリ

config/base_camera_params.jsonのパラメータでBaumer neoAPIを使って画像を撮影する
GUIアプリケーション。撮影枚数とベースディレクトリをUI上で指定できる。
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path

import TkEasyGUI as sg
import cv2

from utils.baumer_camera import BaumerCamera


class ImageCaptureApp:
    def __init__(self):
        self.camera_params = None
        self.is_capturing = False
        self.last_config_path = None
        self.camera = None

    def load_camera_params(self, config_path: str = 'config/base_camera_params.json') -> bool:
        """カメラパラメータをJSONファイルから読み込み"""
        # 同じファイルで変更がない場合は再読み込みしない
        if self.last_config_path == config_path and self.camera_params is not None:
            return True

        if not os.path.exists(config_path):
            sg.popup_error(f"設定ファイルが見つかりません: {config_path}")
            return False

        with open(config_path, 'r', encoding='utf-8') as f:
            self.camera_params = json.load(f)
        self.last_config_path = config_path
        return True

    def capture_images_thread(self, count: int, output_dir: str, window: sg.Window):
        """別スレッドで画像撮影を実行"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        window.post_event('-UPDATE-', {'-UPDATE-': f"{count}枚の画像を撮影開始..."})
        frames = self.camera.continuous_capture(count)

        saved_count = 0
        for i, image in enumerate(frames):
            if not self.is_capturing:
                break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"image_{timestamp}.bmp"
            filepath = os.path.join(output_dir, filename)

            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if cv2.imwrite(filepath, image_bgr):
                saved_count += 1
                progress = int((saved_count / count) * 100)
                window.post_event('-PROGRESS-', {'-PROGRESS-': progress})
                if saved_count % 10 == 0:
                    window.post_event('-UPDATE-', {'-UPDATE-': f"保存進捗: {saved_count}枚完了"})

        window.post_event('-COMPLETE-', {'-COMPLETE-': f"撮影完了: {saved_count}/{count}枚を保存しました\n保存先: {output_dir}"})
        self.is_capturing = False
        window.post_event('-ENABLE-', {'-ENABLE-': True})

    def create_layout(self):
        """GUIレイアウトを作成"""
        return [
            [sg.Text("画像撮影アプリ", font=("Arial", 16, "bold"))],
            [sg.HSeparator()],
            [sg.Text("撮影設定", font=("Arial", 12, "bold"))],
            [sg.Text("撮影枚数:", size=(12, 1)),
             sg.Input(key='-COUNT-', size=(10, 1), default_text="10"),
             sg.Text("枚")],
            [sg.Text("保存ディレクトリ:", size=(12, 1)),
             sg.Input(key='-OUTPUT-', size=(40, 1), default_text="./captured_images"),
             sg.FolderBrowse("参照", size=(8, 1))],
            [sg.HSeparator()],
            [sg.Text("カメラ設定ファイル:", size=(12, 1)),
             sg.Input(key='-CONFIG-', size=(30, 1), default_text="config/base_camera_params.json", readonly=True),
             sg.FileBrowse("参照", size=(8, 1), file_types=(("JSON Files", "*.json"),))],
            [sg.HSeparator()],
            [sg.Button("撮影開始", key='-START-', size=(12, 2), button_color=("white", "green")),
             sg.Button("停止", key='-STOP-', size=(12, 2), button_color=("white", "red"), disabled=True),
             sg.Button("終了", key='-EXIT-', size=(12, 2))],
            [sg.HSeparator()],
            [sg.Text("進捗:", font=("Arial", 10, "bold"))],
            [sg.Canvas(size=(500, 25), key='-PROGRESSBAR-', background_color='white')],
        ]

    def update_progress_bar(self, window: sg.Window, progress: int):
        """プログレスバーを更新"""
        canvas = window['-PROGRESSBAR-']
        canvas.delete("all")

        bar_width, bar_height = 480, 15
        x_offset, y_offset = 10, 5

        canvas.create_rectangle(
            x_offset, y_offset,
            x_offset + bar_width, y_offset + bar_height,
            outline='black', fill='lightgray', width=2
        )

        if progress > 0:
            progress_width = int((progress / 100) * bar_width)
            canvas.create_rectangle(
                x_offset + 1, y_offset + 1,
                x_offset + progress_width - 1, y_offset + bar_height - 1,
                fill='green', outline=''
            )

        canvas.create_text(
            x_offset + bar_width // 2, y_offset + bar_height // 2,
            text=f"{progress}%", font=("Arial", 10), fill='black'
        )

    def _initialize_camera(self) -> bool:
        """カメラを初期化・設定適用"""
        print("カメラを初期化中...")
        self.camera = BaumerCamera()
        self.camera.apply_config(self.camera_params)
        print("カメラの初期化が完了しました")
        return True

    def _cleanup_camera(self):
        """カメラを切断"""
        if self.camera is None:
            return
        if self.camera.cam.IsStreaming():
            self.camera.cam.StopStreaming()
        self.camera.cam.Disconnect()
        self.camera = None
        print("カメラを切断しました")

    def run(self):
        """アプリケーション実行"""
        if not self.load_camera_params():
            return

        if not self._initialize_camera():
            return

        window = sg.Window(
            "画像撮影アプリ",
            self.create_layout(),
            finalize=True,
            resizable=True,
            grab_anywhere=False
        )

        while True:
            event, values = window.read(timeout=100)

            if event == sg.WIN_CLOSED or event == '-EXIT-':
                if self.is_capturing:
                    self.is_capturing = False
                self._cleanup_camera()
                break

            elif event == '-START-':
                # 入力値の検証
                try:
                    count = int(values['-COUNT-'])
                except ValueError:
                    sg.popup_error("撮影枚数は数値を入力してください")
                    continue

                if count <= 0:
                    sg.popup_error("撮影枚数は1以上を入力してください")
                    continue

                output_dir = values['-OUTPUT-']
                if not output_dir:
                    sg.popup_error("保存ディレクトリを指定してください")
                    continue

                config_path = values['-CONFIG-']
                if not os.path.exists(config_path):
                    sg.popup_error(f"設定ファイルが見つかりません: {config_path}")
                    continue

                # 設定ファイルが変更された場合はカメラを再初期化
                if config_path != self.last_config_path:
                    if not self.load_camera_params(config_path):
                        continue
                    self._cleanup_camera()
                    if not self._initialize_camera():
                        continue

                if self.camera is None:
                    sg.popup_error("カメラが初期化されていません")
                    continue

                self.is_capturing = True
                window['-START-'].update(disabled=True)
                window['-STOP-'].update(disabled=False)
                self.update_progress_bar(window, 0)

                threading.Thread(
                    target=self.capture_images_thread,
                    args=(count, output_dir, window),
                    daemon=True
                ).start()

            elif event == '-STOP-':
                self.is_capturing = False
                window['-START-'].update(disabled=False)
                window['-STOP-'].update(disabled=True)

            elif event == '-PROGRESS-':
                progress = values[event] if isinstance(values[event], int) else values[event].get('-PROGRESS-', 0)
                self.update_progress_bar(window, progress)

            elif event == '-COMPLETE-':
                window['-START-'].update(disabled=False)
                window['-STOP-'].update(disabled=True)
                self.update_progress_bar(window, 100)
                message = values[event] if isinstance(values[event], str) else values[event].get('-COMPLETE-', str(values[event]))
                sg.popup_ok("撮影完了", message)

            elif event == '-ERROR-':
                window['-START-'].update(disabled=False)
                window['-STOP-'].update(disabled=True)
                message = values[event] if isinstance(values[event], str) else values[event].get('-ERROR-', str(values[event]))
                sg.popup_error("撮影エラー", message)

            elif event == '-ENABLE-':
                window['-START-'].update(disabled=False)
                window['-STOP-'].update(disabled=True)

        window.close()


def main():
    """メイン処理"""
    app = ImageCaptureApp()
    app.run()


if __name__ == "__main__":
    main()
