import pandas as pd
import os
import json
from datetime import datetime


def save_csv(df: pd.DataFrame, path: str, index: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")