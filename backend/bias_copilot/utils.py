# bias_copilot/utils.py
import re
import numpy as np
from ethnicolr import census_ln  # type: ignore

def infer_gender(username):
    """Mock gender inference based on username length."""
    if not isinstance(username, str):
        username = str(username) if username is not None else ""
    return 0 if len(username) % 2 == 0 else 1  # 0: female (even), 1: male (odd)

def infer_race(username):
    """Infer race from username using ethnicolr."""
    try:
        race = census_ln(username.split()[0])
        return 0 if race in ['black', 'hispanic'] else 1 if race in ['white', 'asian'] else np.random.randint(0, 2)
    except:
        return np.random.randint(0, 2)

def infer_age(text):
    """Infer age from text using regex, handling non-string inputs."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    match = re.search(r'\b(?:age|aged|I\'m|Im)\s*(\d{1,2})\b', text.lower())
    return int(match.group(1)) if match else np.random.randint(18, 65)