import pandas as pd 
import numpy as np
from io import StringIO
import openai
from openai import OpenAI

import dotenv
dotenv.load_dotenv()
api_key=dotenv.get_key('.env', 'api_key')

openai.api_key = api_key

def get_openai_client():
    return OpenAI(api_key=API_KEY)


def parse_csv_response(text):
    try:
        return pd.read_csv(StringIO(text.strip()))
    except Exception as e:
        print("Error parsing CSV:", e)
        print("Raw content:\n", text)
        return pd.DataFrame()
    

def get_standard_constraints(header: str, count: int) -> str:
    Constraints = f"""
    Constraints:
    - Output must be a plain text **CSV** with !!exactly {count}!! rows of data (plus a header row). CHECK IF YOU HAVE THE RIGHT NUMBER OF ROWS
    - **Do not** include any markdown formatting, explanations, column numbering, or extra text
    - All rows must be separated by a single newline (`\\n`)
    - Do not add extra delimiters, trailing commas, or empty lines
    - Use semicolon `;` as the only separator between columns
    - The output must be *strictly machine-readable* — suitable for parsing with pandas.read_csv()
    - All values must be plausible and realistic based on the business context

    - The first words must be the header, exactly as follows:
    {header}
    """
    return Constraints

# This code only exits due to a common gpt problem. If you request x rows, GPT often returns ~90–95% of that, especially for longer, structured prompts.

def parse_and_truncate_csv(text: str, expected_rows: int) -> pd.DataFrame:
    """
    Parses semi-colon-separated CSV text and removes random rows if there are too many.
    Keeps the order of the remaining rows intact.
    """

    # Remove markdown-style prefix lines if present
    lines = [
        line for line in text.strip().splitlines()
        if not line.strip().lower().startswith(("'''", "```"))
    ]

    # Clean formatting: strip whitespace, remove trailing semicolons, skip blanks
    lines = [line.strip().rstrip(";") for line in lines if line.strip()]

    clean_csv = "\n".join(lines)

    try:
        df = pd.read_csv(StringIO(clean_csv), sep=";")
    except Exception as e:
        print("CSV parsing failed:", e)
        print("Raw content:\n", clean_csv)
        return pd.DataFrame()
    
    current_len = len(df)
    
    if current_len > expected_rows:
        # Randomly choose which rows to remove
        num_to_remove = current_len - expected_rows
        drop_indices = np.random.choice(df.index, size=num_to_remove, replace=False)
        df = df.drop(index=drop_indices).reset_index(drop=True)
        return df

    elif current_len == expected_rows:
        return df.reset_index(drop=True)

    else:
        print(f"Only received {current_len} rows, expected {expected_rows}")
        return -1
