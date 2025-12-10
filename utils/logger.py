import logging

from utils.colors import Colors


def log_portfolio_value_change(logger, new_portfolio_value, portfolio_value, new_portfolio_value_prev, log_name,
                               use_color=False, level="info", decimals=2):
    # Helper function to format the value and optionally add color
    def format_value(value):
        if value == 0:
            return f"{value:.{decimals}f}"

        if use_color:
            color = Colors.GREEN if value > 0 else Colors.RED if value < 0 else Colors.RESET
            return f'{color}{value:.{decimals}f}{Colors.RESET}'
        else:
            return f"{value:.{decimals}f}"

    # Format each value with or without color
    new_portfolio_value_str = format_value(new_portfolio_value)
    portfolio_value_str = format_value(portfolio_value)
    portfolio_change_str = format_value(portfolio_value - new_portfolio_value_prev)

    # Construct the log message
    log_message = f"Prev Portfolio Value: {new_portfolio_value_str} | New Portfolio Value: {portfolio_value_str} | Portfolio Change: {portfolio_change_str}"

    # Get the logger method based on the level (e.g., info, debug)
    log_fn = getattr(logger, level, logger.info)

    # Log the message
    log_fn(f"[{log_name}]: {log_message}")

def log_reward(logger, transaction_cost, risk_cost, total_reward, log_name, use_color=False, level="info",
                         decimals=2):


    def format_value(value):
        """Format the value and apply color if needed."""
        if value == 0:
            return f"{value:.{decimals}f}"

        # Apply color if use_color is True
        if use_color:
            color = Colors.GREEN if value > 0 else Colors.RED if value < 0 else Colors.RESET
            return f'{color}{value:.{decimals}f}{Colors.RESET}'
        else:
            return f"{value:.{decimals}f}"

    # Format each value with or without color
    transaction_cost_str = format_value(transaction_cost)
    risk_cost_str = format_value(risk_cost)
    total_reward_str = format_value(total_reward)

    # Create the log message
    log_message = f"Transaction Cost: {transaction_cost_str} | Risk Cost: {risk_cost_str} | Total Reward: {total_reward_str}"

    # Get the logger method based on the level
    log_fn = getattr(logger, level, logger.info)

    # Log the message
    log_fn(f"[{log_name}]: {log_message}")

def log_stock_value(logger, stocks, values, log_name, use_color=False, level="info", decimals=2):
    """Log stock values with optional color coding (green for positive, red for negative)."""
    parts = []

    for stock, value in zip(stocks, values):
        if value == 0:
            continue

        # Decide whether to apply color based on `use_color`
        if use_color:
            color = Colors.GREEN if value > 0 else Colors.RED if value < 0 else Colors.RESET
            parts.append(f'{color}{stock}: {value:.{decimals}f}{Colors.RESET}')
        else:
            # If no color is to be used, just append the stock and value without color
            parts.append(f'{stock}: {value:.{decimals}f}')

    formatted = ', '.join(parts)

    # Get the logger method based on the level
    log_fn = getattr(logger, level, logger.info)

    if formatted:
        log_fn(f"[{log_name}]: {formatted}")


def log_numpy(logger, array, log_name: str, level="info", decimals=2):
    """Log numpy array with specified decimal precision in one line."""
    formatted = '[' + ', '.join(f'{x:.{decimals}f}' for x in array) + ']'
    log_fn = getattr(logger, level, logger.info)
    log_fn(f"[{log_name}]: {formatted}")


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

        # Console handler with colors
        console_handler = logging.StreamHandler()
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
