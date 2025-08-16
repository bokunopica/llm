import os
import sys
import atexit
from datetime import datetime
from typing import Optional

class Tee:
    """让 stdout/stderr 同时写到文件和终端"""

    def __init__(self, filename: str, mode: str = "w"):
        # 行缓冲，便于实时落盘
        self.file = open(filename, mode, buffering=1)
        self.stdout = sys.stdout
        self.filename = filename

    def write(self, data: str):
        self.stdout.write(data)  # 打印到终端
        self.file.write(data)    # 写到文件

    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        """关闭文件句柄"""
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()


# 全局变量跟踪当前的 Tee 对象
_current_tee: Optional[Tee] = None
_original_stdout = sys.stdout
_original_stderr = sys.stderr


def setup_logging(log_dir="logs"):
    """初始化双输出日志"""
    global _current_tee
    
    # 如果已经有活跃的 Tee，先关闭它
    cleanup_logging()
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")
    
    _current_tee = Tee(log_file)
    sys.stdout = _current_tee
    sys.stderr = _current_tee
    
    print(f"日志已启用，保存到: {log_file}")
    return log_file


def cleanup_logging():
    """清理日志设置，恢复原始的 stdout/stderr"""
    global _current_tee
    
    if _current_tee is not None:
        # 恢复原始的 stdout/stderr
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr
        
        # 关闭当前的 Tee
        _current_tee.close()
        _current_tee = None
        
        print("日志已关闭，恢复到标准输出")


def get_current_log_file() -> Optional[str]:
    """获取当前日志文件路径"""
    if _current_tee is not None:
        return _current_tee.filename
    return None


# 注册程序退出时的清理函数
atexit.register(cleanup_logging)


# 上下文管理器版本
class LoggingContext:
    """日志上下文管理器，确保正确的资源管理"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.log_file = None
    
    def __enter__(self):
        self.log_file = setup_logging(self.log_dir)
        return self.log_file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_logging()


# 装饰器版本
def with_logging(log_dir="logs"):
    """装饰器：为函数添加日志记录"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            log_file = setup_logging(log_dir)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                cleanup_logging()
        return wrapper
    return decorator