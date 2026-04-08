import re


def clean(text):
    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"[^0-9.]", "", text)

    if text.count(".") > 1:
        first = text.find(".")
        text = text[:first + 1] + text[first + 1:].replace(".", "")

    if text.startswith("."):
        text = text[1:]

    if len(text) == 0:
        return ""

    return text


def enforce_format(text, max_len=12):
    text = clean(text)

    if len(text) > max_len:
        text = text[:max_len]

    return text


def confidence_filter(text, min_len=1):
    if len(text) < min_len:
        return ""

    return text


def postprocess(text):
    text = enforce_format(text)
    text = confidence_filter(text)
    return text

def apply_confidence(text, conf, threshold=0.6):
    if conf < threshold:
        return ""
    return text