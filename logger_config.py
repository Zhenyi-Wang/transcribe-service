import os
import logging
from pathlib import Path

# 创建不缓冲的文件处理器
class UnbufferedFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.stream.flush()

def setup_logger(name: str = None, log_file: str = None, level: int = logging.INFO):
    """设置并返回一个配置好的logger

    Args:
        name: logger名称，默认使用调用模块的名称
        log_file: 日志文件名，默认为模块名.log
        level: 日志级别，默认为INFO

    Returns:
        logging.Logger: 配置好的logger
    """
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 获取logger
    if name is None:
        # 自动获取调用模块的名称
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'default')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 设置日志文件名
    if log_file is None:
        log_file = f"{name.split('.')[-1]}.log"

    # 创建文件处理器
    file_handler = UnbufferedFileHandler(log_dir / log_file, encoding='utf-8')
    file_handler.setLevel(level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 防止日志重复
    logger.propagate = False

    return logger

# 创建默认logger
default_logger = setup_logger('transcribe_service')