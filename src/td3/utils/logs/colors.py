"""
Terminal color and styling constants for the TD3 training pipeline.

Provides ANSI escape codes via the Colors class for formatted
console output during training and logging.

Author: Peter Likavec
"""


class Colors:
    """
    ANSI escape code constants for terminal text styling.
    """

    WHITE = '\033[97m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
