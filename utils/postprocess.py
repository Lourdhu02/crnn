import re

def clean(text):
    text = re.sub(r"[^0-9\.]", "", text)
    if text.count(".") > 1:
        parts = text.split(".")
        text = parts[0] + "." + "".join(parts[1:])
    return text
