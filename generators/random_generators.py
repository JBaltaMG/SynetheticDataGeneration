import pandas as pd
import numpy as np
import random
import utils.utils as utils

def primary_key():
    return np.random.choice(range(1000, 9999), size=1)[0]

def generate_employee_ids(n):
    return np.random.choice(range(100, 999), size=n, replace=False)

def generate_customer_ids(n):
    return list(range(100, 100 + n))

def generate_document_number_series(count):
    return [f"78{str(i).zfill(6)}" for i in range(1, count + 1)]

def generate_account_numbers(n=50):
    # More realistic account numbers based on pattern
    return sorted(np.random.choice(
        [i for i in range(10000, 99999) if str(i).startswith(('1', '4', '5'))],
        size=n,
        replace=False
    ))

def generate_month_list(year_start: int = 2025, year_end: int = 2025) -> list:
    return [utils.format_month_string(year, m) for year in range(year_start, year_end + 1) for m in range(1, 13)]

def generate_dim_date(year_start: int = 2025, year_end: int = 2025) -> pd.DataFrame:

    dates = pd.date_range(start=f"{year_start}-01-01", end=f"{year_end}-12-31", freq="D")

    return pd.DataFrame({
        "date_id": range(1, len(dates) + 1),
        "date": dates,
        "month": dates.month,
        "quarter": dates.quarter,
        "year": dates.year
    })

def generate_document_metadata(n: int, start_index: int = 0) -> pd.DataFrame:
    doc_types = ["Invoice", "CreditNote", "Adjustment", "Transfer"]
    currencies = ["DKK", "EUR"]

    document_numbers = [
        f"78{start_index + i:06d}"  # Ensures no overlap and fixed width
        for i in range(n)
    ]

    return pd.DataFrame({
        "document_number": document_numbers,
        "document_type": random.choices(doc_types, k=n),
        "currency": random.choices(currencies, k=n)
    })


def generate_account_ids(account_types: pd.Series) -> pd.Series:
    """
    Generate 5-digit account numbers based on account type prefix:
    - Asset: starts with 1
    - Equity: starts with 2
    - Revenue: starts with 3
    - Product Expense: starts with 4
    - Service Expense: starts with 5
    - Payroll Expense: starts with 6
    """
    prefix_map = {
        'Asset': '1',
        'Equity': '2',
        'Revenue': '3',
        'Product Expense': '4',
        'Service Expense': '5',
        'Payroll Expense': '6',
    }

    used_numbers = set()
    account_numbers = []

    for acc_type in account_types:
        prefix = prefix_map.get(acc_type, '9')  # fallback to '9' for unknown types
        while True:
            # Generate a 5-digit number starting with the prefix
            num = int(prefix + ''.join(np.random.choice(list('0123456789'), size=4)))
            if num not in used_numbers:
                used_numbers.add(num)
                account_numbers.append(num)
                break

    return pd.Series(account_numbers, index=account_types.index)