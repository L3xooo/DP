"""
Logging and formatting utilities for the TD3 training pipeline.

Provides colored terminal output for numeric values and a custom
logging formatter that applies ANSI colors based on log level.

Author: Peter Likavec
"""

from td3.utils.logs.colors import Colors
import logging


def format_number_value(value, decimals=2, use_color=False):
    """Format a numeric value as a fixed-decimal string with optional color."""
    if value == 0:
        return f"{value:.{decimals}f}"

    if use_color:
        color = Colors.GREEN if value > 0 else Colors.RED if value < 0 else Colors.RESET
        return f'{color}{value:.{decimals}f}{Colors.RESET}'
    else:
        return f"{value:.{decimals}f}"


class ColoredFormatter(logging.Formatter):
    """
    Custom log formatter that colorizes output based on log level.

    Applies ANSI colors to each log record: cyan for DEBUG, default for INFO,
    yellow for WARNING, red for ERROR, and bold red for CRITICAL.
    """

    FORMATS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.RESET,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }

    def format(self, record):
        """Apply level-appropriate color to the log record and format it."""
        log_color = self.FORMATS.get(record.levelno, Colors.RESET)
        formatter = logging.Formatter(
            f"{log_color}%(asctime)s - %(name)s - %(levelname)s - %(message)s{Colors.RESET}"
        )
        return formatter.format(record)
