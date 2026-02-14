import logging
import os
import sys
from logging.handlers import RotatingFileHandler


_RESET = "\033[0m"
_COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
}


class ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str | None = None, use_color: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not self.use_color:
            return formatted

        color = _COLORS.get(record.levelname)
        if not color:
            return formatted
        return f"{color}{formatted}{_RESET}"


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return True


def _parse_log_level(level_name: str | None) -> int:
    candidate = (level_name or os.getenv("LOG_LEVEL", "INFO")).upper().strip()
    return getattr(logging, candidate, logging.INFO)


def setup_logging(app_name: str = "meeting-transcriptions", log_dir: str = "output/logs", level_name: str | None = None) -> None:
    root_logger = logging.getLogger()
    if getattr(root_logger, "_meeting_logging_configured", False):
        return

    os.makedirs(log_dir, exist_ok=True)

    level = _parse_log_level(level_name)
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        ColorFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            use_color=_supports_color(),
        )
    )

    max_mb_raw = os.getenv("LOG_FILE_MAX_MB", "5")
    backup_raw = os.getenv("LOG_FILE_BACKUP_COUNT", "5")
    try:
        max_bytes = int(float(max_mb_raw) * 1024 * 1024)
    except ValueError:
        max_bytes = 5 * 1024 * 1024
    try:
        backup_count = int(backup_raw)
    except ValueError:
        backup_count = 5

    log_file_path = os.path.join(log_dir, f"{app_name}.log")
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    logging.captureWarnings(True)

    setattr(root_logger, "_meeting_logging_configured", True)
    logging.getLogger(app_name).info(
        "Logging initialized (level=%s, file=%s)",
        logging.getLevelName(level),
        log_file_path,
    )
