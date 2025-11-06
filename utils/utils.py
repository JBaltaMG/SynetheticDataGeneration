import random
import numpy as np
import pandas as pd
import os
import shutil

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def read_text(path: str, max_chars: int = None, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        text = f.read()
    if max_chars is not None:
        return text[:max_chars]
    return text

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

def sample_employees(count_employee: int, filename: str = "data/inputdata/NameList.csv", if_fullname: bool = False) -> pd.DataFrame:
    """
    Samples a number of employees from a CSV with 'first_name' and 'last_name',
    and returns a DataFrame with first_name, last_name, and full name.

    Args:
        count_employee (int): Number of employee names to sample.
        filename (str): Path to the name list CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns 'first_name', 'last_name', 'EmployeeName'.
    """
    # Load name list
    name_df = pd.read_csv(filename)

    # Check required columns
    if not {'first_name', 'last_name'}.issubset(name_df.columns):
        raise ValueError("CSV must contain 'first_name' and 'last_name' columns.")

    # Sample rows (with replacement if not enough unique)
    if count_employee > len(name_df):
        sampled_df = name_df.sample(n=count_employee, replace=True).reset_index(drop=True)
    else:
        sampled_df = name_df.sample(n=count_employee, replace=False).reset_index(drop=True)

    # Add full name column
    if if_fullname:
        sampled_df['EmployeeName'] = sampled_df['first_name'].str.strip() + ' ' + sampled_df['last_name'].str.strip()
        return sampled_df[['first_name', 'last_name', 'EmployeeName']]
    
    return sampled_df[['first_name', 'last_name']]


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
    df_vendors: pd.DataFrame,
    name_column="product_name"
    ) -> pd.DataFrame:
    """
    Maps each item (product/service/employee/procurement) to multiple GL accounts and departments.

    Special case:
    - If name_column == 'employee_id', maps each employee to payroll accounts.

    Returns a DataFrame with one row per (item, account_id, Department) combination.
    """
    mappings = []
    
    if name_column == "product_name":
        expense_accounts = df_accounts[df_accounts["account_type"] == "Revenue"]
    elif name_column == "service_name":
        expense_accounts = df_accounts[df_accounts["account_type"] == "Service Expense"]
    elif name_column == "procurement_name":
        expense_accounts = df_accounts[df_accounts["account_type"] == "Product Expense"]
    else:
        raise ValueError(f"Unsupported name_column: {name_column}")

    for _, row in df.iterrows():
        item_value = row["name"]
        n_accounts_per_item = np.random.randint(2, 7)
        gl_sample = expense_accounts.sample(n_accounts_per_item, replace=True)

        for _, acc in gl_sample.iterrows():
            mapping = {
                "name": item_value,
                "account_id": acc["account_id"],
                "account_name": acc["name"],
            }

            if name_column == "product_name" and df_customers is not None:
                customer = df_customers.sample(1).iloc[0]
                mapping["customer_name"] = customer["name"]

            if name_column == "service_name" and df_vendors is not None:
                vendor = df_vendors.sample(1).iloc[0]
                mapping["vendor_name"] = vendor["name"]

            if name_column == "procurement_name" and df_vendors is not None:
                vendor = df_vendors.sample(1).iloc[0]
                mapping["vendor_name"] = vendor["name"]

            mappings.append(mapping)

    return pd.DataFrame(mappings)
 

def assign_departments(df_pay: pd.DataFrame, df_departments: pd.DataFrame) -> pd.DataFrame:
    """
    Assign departments to employees based on proportionality.

    Args:
        df_pay: DataFrame with at least an 'employee_id' column.
        df_departments: DataFrame with 'department_name' and 'proportionality' columns.
        seed: Random seed for reproducibility.

    Returns:
        df_pay with a new 'department_name' column assigned.
    """
    n_employees = len(df_pay)

    # Normalize proportionality to ensure sum = 1
    df_departments['proportionality'] = df_departments['proportionality'] / df_departments['proportionality'].sum()

    # Determine how many employees per department
    df_departments['num_employee'] = (df_departments['proportionality'] * n_employees).round().astype(int)

    # Correct for rounding errors to make total match
    diff = n_employees - df_departments['num_employee'].sum()
    if diff != 0:
        # Add or subtract the diff to the department with the largest proportion
        idx = df_departments['proportionality'].idxmax()
        df_departments.loc[idx, 'num_employee'] += diff

    # Build the list of departments to assign
    department_assignments = []
    for _, row in df_departments.iterrows():
        department_assignments.extend([row['name']] * row['num_employee'])

    np.random.shuffle(department_assignments)

    # Assign to employees
    df_pay = df_pay.copy()
    df_pay['name'] = department_assignments

    cols = ["role_name", "monthly_pay", "first_name", "last_name", "employee_id", "department_id"]

    df_pay = df_pay.rename(columns={
        "role_name": "role_name",
        "monthly_pay": "monthly_pay",
        "first_name": "first_name",
        "last_name": "last_name",
        "employee_id": "employee_id",
        "name": "department_id"
    })
    
    return df_pay[cols]

def mirror_intercompany_flows(df_revenue: pd.DataFrame,
                              df_cogs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create mirrored intercompany entries so that total intercompany balances net to zero.

    Parameters
    ----------
    df_rev_ic : pd.DataFrame
        Intercompany revenue lines (e.g., account_id = 4007).
    df_cogs_ic : pd.DataFrame
        Intercompany COGS lines (e.g., account_id = 4009).

    Returns
    -------
    rev_mirror : pd.DataFrame
        New revenue lines created from intercompany COGS (4009 -> 4007).
    cogs_mirror : pd.DataFrame
        New COGS lines created from intercompany revenue (4007 -> 4009).
    """
    df_rev_ic  = df_revenue.query("account_id == 4007").copy()
    df_cogs_ic = df_cogs.query("account_id == 4009").copy()

    # --- Helper: swap BU <-> Party columns ---
    def swap_bu_and_party(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["bu_id"], df["party_id"] = df["party_id"], df["bu_id"]
        df["bu_name"], df["party_name"] = df["party_name"], df["bu_name"]
        return df

    # --- Revenue → mirror into COGS ---
    cogs_mirror = df_rev_ic.copy()
    cogs_mirror = swap_bu_and_party(cogs_mirror)
    cogs_mirror["account_id"] = 4009
    cogs_mirror["category"] = "COGS"
    
    # --- COGS → mirror into Revenue ---
    rev_mirror = df_cogs_ic.copy()
    rev_mirror = swap_bu_and_party(rev_mirror)
    rev_mirror["account_id"] = 4007
    rev_mirror["category"] = "Revenue"

    df_revenue_final = pd.concat([df_revenue, rev_mirror], ignore_index=True)
    df_cogs_final    = pd.concat([df_cogs, cogs_mirror], ignore_index=True)
    
    return df_revenue_final, df_cogs_final

def get_party_list(
    df_fact: pd.DataFrame,
    df_parties: pd.DataFrame,
    df_bus: pd.DataFrame,
    name: str = "party"
) -> pd.DataFrame:  
    
    df_list = df_fact[["party_id", "party_name"]].drop_duplicates("party_id")
    df_list = df_list.merge(df_parties, left_on="party_id", right_on="party_ID", how="left")
    df_list = df_list.drop(columns=["party_name_y", "party_ID"])
    df_list = df_list.rename(columns={"party_name_x": "party_name"})

    mask_nan = df_list['party_country'].isna()

    # Merge on matching names (party_name ↔ bu_name)
    df_list.loc[mask_nan, 'party_country'] = (
        df_list.loc[mask_nan, 'party_name']
        .map(df_bus.set_index('bu_name')['country'])
    )

    df_list.loc[mask_nan, 'party_type'] = 'INTERNAL_BU'

    # === 3. Dynamic renaming ===
    rename_dict = {
        "party_id": f"{name}_id",
        "party_name": f"{name}_name",
        "party_type": f"{name}_type",
        "party_country": f"{name}_country",
    }

    df_list = df_list.rename(columns=rename_dict)
    df_terms = pd.read_csv("data/inputdata/generic_terms.csv", sep=",")
    df_list["terms_id"] = pd.Series(np.random.choice(df_terms["terms_id"], size=len(df_list), replace=True)).reset_index(drop=True)

    return df_list
