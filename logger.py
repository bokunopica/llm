
import os
import sys
from datetime import datetime

class Tee:
    """让 stdout/stderr 同时写到文件和终端"""

    def __init__(self, filename: str, mode: str = "w"):
        # 行缓冲，便于实时落盘
        self.file = open(filename, mode, buffering=1)
        self.stdout = sys.stdout

    def write(self, data: str):
        self.stdout.write(data)  # 打印到终端
        self.file.write(data)    # 写到文件

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def setup_logging(log_dir="logs"):
    """初始化双输出日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")
    tee = Tee(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return log_file