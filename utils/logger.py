import time
from loguru import logger

class SystemLogger:
    def __init__(self, log_file: str = "cam_security.log"):
        self.logger = logger
        self.logger.add(log_file, rotation="10 MB", retention="7 days")

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def sleep(self, seconds: float, message: str = None):
        if message:
            self.logger.info(message)
        time.sleep(seconds)

sys_logger = SystemLogger()
