import re


def clean(text):
    if text is None:
        return ""
    text = str(text)
    # keep only digits and dot
    text = re.sub(r"[^0-9.]", "", text)
    # keep only first decimal point
    if text.count(".") > 1:
        first = text.find(".")
        text = text[: first + 1] + text[first + 1:].replace(".", "")
    return text


def enforce_format(text, max_len=12):
    text = clean(text)
    if len(text) > max_len:
        text = text[:max_len]
    return text


def postprocess(text):
    # raw meter reading: keep leading zeros, keep as displayed
    return enforce_format(text)


def apply_confidence(text, conf, threshold=0.6):
    if conf < threshold:
        return ""
    return text
