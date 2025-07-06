import os
import time
import subprocess

file_path = "output/relationship_graph.png"


def get_modified_time(path):
    return os.path.getmtime(path) if os.path.exists(path) else None


print("👀 グラフ画像の更新を監視します（Ctrl+Cで停止）")

last_modified = get_modified_time(file_path)

while True:
    time.sleep(1)
    current_modified = get_modified_time(file_path)

    if current_modified is None:
        continue

    if current_modified != last_modified:
        last_modified = current_modified
        print("📂 グラフが更新されました。画像を表示します。")
        subprocess.run(["open", file_path])
