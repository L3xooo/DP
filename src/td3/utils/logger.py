import logging
import sys

from td3.utils.colors import Colors
from td3.utils.formater import format_number_value, ColoredFormatter


def log_values_with_color(
    logger, values: dict, use_color=False, level="info", log_name=None
):
    formatted_items = []

    for key, value in values.items():
        try:
            colored_value = format_number_value(value, use_color=use_color)
        except Exception:
            colored_value = str(value)
        formatted_items.append(f"{key}: {colored_value}")

    message = " | ".join(formatted_items)
    if log_name:
        message = f"[{log_name}]: {message}"
    log_fn = getattr(logger, level, logger.info)
    log_fn(message)


def log_stock_value(
    logger, stocks, values, log_name, use_color=False, level="info", decimals=2
):
    """Log stock values with optional color coding (green for positive, red for negative, white for zero)."""
    parts = []

    for stock, value in zip(stocks, values):
        # Decide the color based on the value
        if value == 0:
            continue
            color = Colors.WHITE  # For zero value, we use white
        elif value > 0:
            color = Colors.GREEN  # Green for positive values
        else:
            color = Colors.RED  # Red for negative values

        # Decide whether to apply color based on `use_color`
        if use_color:
            parts.append(f'{color}{stock}: {value:.{decimals}f}{Colors.RESET}')
        else:
            # If no color is to be used, just append the stock and value without color
            parts.append(f'{stock}: {value:.{decimals}f}')

    formatted = ', '.join(parts)
    log_fn = getattr(logger, level, logger.info)
    log_fn(f"[{log_name}]: {formatted}")

    # if formatted:
    #     log_fn(f"[{log_name}]: {formatted}")


class LoggerFactory:
    LOG_FILE = "app.log"

    @staticmethod
    def clear_log():
        open(LoggerFactory.LOG_FILE, 'w').close()

    @staticmethod
    def create_logger(name: str):
        LoggerFactory.clear_log()
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.disabled = True

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        # File handler without colors (plain text)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(LoggerFactory.LOG_FILE)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger


class WithLogger:
    """Decorator class that attaches a logger to any class."""

    def __call__(self, cls):
        # Attach logger based on class name
        cls.logger = LoggerFactory.create_logger(cls.__name__)
        return cls
