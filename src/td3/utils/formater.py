from td3.utils.colors import Colors
import logging


def format_number_value(value, decimals=2, use_color=False):
    if value == 0:
        return f"{value:.{decimals}f}"

    if use_color:
        color = (
            Colors.GREEN
            if value > 0
            else Colors.RED
            if value < 0
            else Colors.RESET
        )
        return f'{color}{value:.{decimals}f}{Colors.RESET}'
    else:
        return f"{value:.{decimals}f}"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors based on log level."""

    FORMATS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.RESET,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }

    def format(self, record):
        log_color = self.FORMATS.get(record.levelno, Colors.RESET)
        formatter = logging.Formatter(
            f"{log_color}%(asctime)s - %(name)s - %(levelname)s - %(message)s{Colors.RESET}"
        )
        return formatter.format(record)
