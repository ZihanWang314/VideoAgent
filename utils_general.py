import logging
import pickle
from typing import Optional
import json
import random
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r"\{.*?\}|\[.*?\]"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        print("No valid JSON found in the text.")
        return None


def parse_text_find_number(text):
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        return match if match in range(-1, 5) else random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        return -1


def parse_text_find_confidence(text):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        return match if match in range(1, 4) else random.randint(1, 3)
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1
