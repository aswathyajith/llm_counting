import numpy as np
import re


def extract_model_count(x):
    # print(x)
    text = x["model_response_text"]
    # match for pattern <count>?</count>
    pattern = r"<count>(\d+)</count>"
    try:
        match = re.search(pattern, text)
        if match:
            model_count = int(match.group(1))
        else:
            model_count = np.nan
    except Exception:
        model_count = np.nan
    x["model_count"] = model_count
    return x
