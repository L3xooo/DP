from utils.colors import Colors


def format_number_value(value, decimals=2, use_color=False):
    if value == 0:
        return f"{value:.{decimals}f}"

    if use_color:
        color = Colors.GREEN if value > 0 else Colors.RED if value < 0 else Colors.RESET
        return f'{color}{value:.{decimals}f}{Colors.RESET}'
    else:
        return f"{value:.{decimals}f}"