# logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

# Имя для нашего логгера – можно любое, но оно должно быть уникальным для приложения
LOGGER_NAME = "meeting_agent"
LOG_DIR = "logs"                     # директория рядом с logger.py или абсолютный путь
LOG_FILE = os.path.join(LOG_DIR, "app.log")
MAX_BYTES = 5 * 1024 * 1024         # 5 МБ на файл
BACKUP_COUNT = 3                     # хранить 3 старых файла

# Уровень логирования по умолчанию (можно переопределить через переменную окружения)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logger() -> logging.Logger:
    """Создаёт и настраивает единственный экземпляр логгера для всего приложения."""
    logger = logging.getLogger(LOGGER_NAME)

    # Чтобы не добавлять обработчики повторно при повторных вызовах
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    # Создаём папку для логов, если её нет
    os.makedirs(LOG_DIR, exist_ok=True)

    # Файловый обработчик с ротацией (чтобы не разрастался бесконечно)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(LOG_LEVEL)

    # Консольный обработчик (чтобы видеть логи и в терминале)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)

    # Формат: время, уровень, модуль, сообщение
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# При импорте модуля сразу создаём глобальный объект логгера
logger = setup_logger()