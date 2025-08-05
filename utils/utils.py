import random
import numpy as np
import pandas as pd
import os
import shutil

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def add_id_column(df: pd.DataFrame, column_name: str = "ID", start_index: int = 1) -> pd.DataFrame:
    """
    Adds a sequential ID column to the DataFrame.
    
    """
    if column_name in df.columns:
        raise ValueError(f"Column '{column_name}' already exists in DataFrame.")

    df[column_name] = range(start_index, start_index + len(df))
    return df

def create_clean_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # Deletes the entire folder and contents
    os.makedirs(path)        # Creates a fresh empty folder    

def simulate_log_normal_pay_distribution(n: int, scale: float = 50000, top_multiplier: float = 3, seed: int = 42) -> pd.Series:
    """
    Simulates a skewed pay distribution:
    - log-normal shape
    - controlled top-heavy distribution
    - consistent across runs
    """
    np.random.seed(seed)
    log_mean = 0.7       # controls central mass (adjust up to raise median)
    log_sigma = 0.6      # controls spread (adjust up to increase top/bottom gap)

    # Generate log-normal and sort descending
    raw = np.random.lognormal(mean=log_mean, sigma=log_sigma, size=n)
    raw_sorted = np.sort(raw)[::-1]

    # Apply scale + top-end inflation
    adjusted = raw_sorted / raw_sorted.max()  # normalize to [0, 1]
    inflated = adjusted ** 0.75               # apply nonlinear inflation
    pay_values = inflated * scale * top_multiplier

    return pd.Series(np.round(pay_values, -2))  # round to nearest 100

def sample_employees(count_employees: int, filename: str = "data/inputdata/NameList.csv", if_fullname: bool = False) -> pd.DataFrame:
    """
    Samples a number of employees from a CSV with 'FirstName' and 'LastName',
    and returns a DataFrame with FirstName, LastName, and full name.

    Args:
        count_employees (int): Number of employee names to sample.
        filename (str): Path to the name list CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns 'FirstName', 'LastName', 'EmployeeName'.
    """
    # Load name list
    name_df = pd.read_csv(filename)

    # Check required columns
    if not {'FirstName', 'LastName'}.issubset(name_df.columns):
        raise ValueError("CSV must contain 'FirstName' and 'LastName' columns.")

    # Sample rows (with replacement if not enough unique)
    if count_employees > len(name_df):
        sampled_df = name_df.sample(n=count_employees, replace=True).reset_index(drop=True)
    else:
        sampled_df = name_df.sample(n=count_employees, replace=False).reset_index(drop=True)

    # Add full name column
    if if_fullname:
        sampled_df['EmployeeName'] = sampled_df['FirstName'].str.strip() + ' ' + sampled_df['LastName'].str.strip()
        return sampled_df[['FirstName', 'LastName', 'EmployeeName']]
    
    return sampled_df[['FirstName', 'LastName']]


def format_month_string(year: int, month: int) -> str:
    return f"{year}-{month:02d}"

def convert_column_to_percentage(df: pd.DataFrame, column: str, scale: float = 1.0) -> pd.DataFrame:
    """
    Converts a numeric column into percentages (relative to column sum), and replaces the original column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): Column to convert to percentages.
        scale (float): Scaling factor (1.0 for fractions, 100.0 for percent values)

    Returns:
        pd.DataFrame: Updated DataFrame with original column replaced by percentage values.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    total = df[column].sum()
    if total == 0:
        raise ValueError(f"Column '{column}' has a total sum of 0 — cannot convert to percentage.")

    df[column] = df[column] / total * scale
    return df

def create_mapping_from_metadata(
    df: pd.DataFrame, 
    df_accounts: pd.DataFrame, 
    df_departments: pd.DataFrame, 
    df_customers: pd.DataFrame, 
    name_column="ProductName"
) -> pd.DataFrame:
    """
    Maps each item (product/service/employee/procurement) to multiple GL accounts and departments.

    Special case:
    - If name_column == 'Employee_ID', maps each employee to payroll accounts.

    Returns a DataFrame with one row per (item, GLAccount, Department) combination.
    """
    mappings = []

    if name_column == "ProductName":
        expense_accounts = df_accounts[df_accounts["AccountType"] == "Revenue"]
    elif name_column == "ServiceName":
        expense_accounts = df_accounts[df_accounts["AccountType"] == "Service Expense"]
    elif name_column == "Employee_ID":
        expense_accounts = df_accounts[df_accounts["AccountType"] == "Payroll"]
    elif name_column == "ProcurementName":
        expense_accounts = df_accounts[df_accounts["AccountType"] == "Product Expense"]
    else:
        raise ValueError(f"Unsupported name_column: {name_column}")

    for _, row in df.iterrows():
        item_value = row["name"]
        n_accounts_per_item = np.random.randint(2, 7)
        gl_sample = expense_accounts.sample(n_accounts_per_item, replace=True)

        for _, acc in gl_sample.iterrows():
            mapping = {
                "name": item_value,
                "GLAccount": acc["Account_ID"],
                "GLAccountName": acc["name"],
            }

            if name_column == "ProductName" and df_customers is not None:
                customer = df_customers.sample(1).iloc[0]
                mapping["CustomerName"] = customer["name"]

            mappings.append(mapping)

    return pd.DataFrame(mappings)