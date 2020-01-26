import time

def make_loading_bar(length, fraction):
    filled_length = int(round(fraction * length))
    filled = ["="] * filled_length
    if len(filled) > 0 and fraction != 1.0:
        filled[-1] = ">"
    filled = "".join(filled)
    loading_bar = f"[{filled}{(length - filled_length) * ' '}]"
    return loading_bar

def readable_time(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:5.2f}h"
    elif seconds >= 60:
        return f"{seconds / 60:5.2f}m"
    else:
        return f"{seconds:5.2f}s"

def eta(time, fraction_done):
    total_time = time / fraction_done if fraction_done != 0 else float("inf")
    eta = total_time - time
    return eta
