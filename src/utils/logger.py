from loguru import logger
import os

os.makedirs("logs", exist_ok=True)

logger.add(
    "logs/aml_kyc_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"
)

def get_logger(name: str = "aml_kyc"):
    return logger.bind(module=name)