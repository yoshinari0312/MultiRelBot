import os
from datetime import datetime

# タイムスタンプ付きディレクトリを作成
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_ROOT = os.path.join("static", "logs", timestamp_str)
os.makedirs(LOG_ROOT, exist_ok=True)
