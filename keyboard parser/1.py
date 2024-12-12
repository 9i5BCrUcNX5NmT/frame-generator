import pandas as pd
from pynput import keyboard
import os
import datetime
import threading
import time


class KeyLogger:
    def __init__(self, interval=10, file_path="../data/keys/key_log.csv"):
        self.data = []
        self.lock = threading.Lock()
        self.interval = interval
        self.file_path = file_path  # Путь к файлу
        self.writer_thread = threading.Thread(target=self.write_data_periodically)
        self.writer_thread.daemon = True

    def on_press(self, key):
        try:
            self.data.append(
                {
                    "key": key.char,
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        except AttributeError:
            self.data.append(
                {
                    "key": str(key),
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    def on_release(self, key):
        if key == keyboard.Key.esc:
            # Записываем оставшиеся данные перед выходом
            self.save_data()
            return False

    def write_data_periodically(self):
        while True:
            with self.lock:
                if self.data:
                    df = pd.DataFrame(self.data)
                    if os.path.exists(self.file_path):
                        existing_df = pd.read_csv(self.file_path)
                        combined_df = pd.concat([existing_df, df])
                        combined_df.to_csv(self.file_path, index=False)
                    else:
                        df.to_csv(self.file_path, index=False)
                    self.data = []
            time.sleep(self.interval)

    def save_data(self):
        with self.lock:
            if self.data:
                df = pd.DataFrame(self.data)
                if os.path.exists(self.file_path):
                    existing_df = pd.read_csv(self.file_path)
                    combined_df = pd.concat([existing_df, df])
                    combined_df.to_csv(self.file_path, index=False)
                else:
                    df.to_csv(self.file_path, index=False)
                self.data = []

    def start(self):
        self.writer_thread.start()
        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            listener.join()


if __name__ == "__main__":
    key_logger = KeyLogger(interval=10)  # запись данных в файл каждые 10 секунд
    key_logger.start()
