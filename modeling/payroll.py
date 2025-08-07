import pandas as pd
import numpy as np
import random
import generators.random_generators as random_generators
from modeling import mapping
import utils.utils as utils

utils.set_global_seed(42)

def create_pay_roll(df_roles: pd.DataFrame, df_employees: pd.DataFrame, mean_pay: float = 42000, if_long: bool = True) -> pd.DataFrame:
    df_role = assign_monthly_pay(df_roles, mean_pay)
    df = pd.concat([df_role, df_employees], axis=1)    
    df["employee_id"] = random_generators.generate_employee_ids(len(df))
    
    if if_long:
        df = expand_employees_across_months(df)
        df = apply_raise_and_fire(df)
    else:
        df = df.copy()
        
    return df

def assign_monthly_pay(df_roles: pd.DataFrame, mean_pay: float = 42000) -> pd.DataFrame:
    pay = utils.simulate_log_normal_pay_distribution(len(df_roles), scale=mean_pay)
    df_roles["monthly_pay"] = pay
    return df_roles

def expand_employees_across_months(df_employees: pd.DataFrame, year: int = 2025) -> pd.DataFrame:
    months = random_generators.generate_month_list(year_start=year-5, year_end=year)
    rows = []

    for _, row in df_employees.iterrows():
        for m in months:
            rows.append({
                "employee_id": row["employee_id"],
                "role_name": row["role_name"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "month": m,
                "monthly_pay": row["monthly_pay"]
            })

    return pd.DataFrame(rows)

def apply_raise_and_fire(df: pd.DataFrame, raise_pct: float = 0.05, fire_pct: float = 0.03) -> pd.DataFrame:
    employee_ids = df['employee_id'].unique()

    raise_ids = random.sample(list(employee_ids), int(len(employee_ids) * raise_pct))
    remaining = list(set(employee_ids) - set(raise_ids))
    fire_ids = random.sample(remaining, int(len(employee_ids) * fire_pct))
    for eid in raise_ids:
        start = random.choice(df["month"].unique())
        df.loc[(df["employee_id"] == eid) & (df["month"] >= start), "monthly_pay"] *= 1.05

    for eid in fire_ids:
        start = random.choice(df["month"].unique())
        df.loc[(df["employee_id"] == eid) & (df["month"] >= start), "monthly_pay"] = 0

    return df

def add_taxes(df_payroll: pd.DataFrame) -> pd.DataFrame:

    df_payroll = df_payroll.rename(columns={'monthly_pay': '0013'})

    # Tax configuration
    tax_config = {
        "A-skat": {"code": "0015", "type": "percent", "rate": 0.30, "source": "0013"},
        "AM-bidrag": {"code": "0016", "type": "percent", "rate": 0.08, "source": "0013"},
        "Fri bil": {"code": "0019", "type": "percent", "rate": 0.06, "source": "0013"},
        "Fri telefon": {"code": "0020", "type": "fixed", "value": 275},
        "Sundhedsforsikring": {"code": "0026", "type": "fixed", "value": 44},
        "ATP-bidrag": {"code": "0046", "type": "fixed", "value": 297},
        "Pension medarbejder": {"code": "0147", "type": "percent", "rate": 0.03, "source": "0013"},
        "Pension arbejdsgiver": {"code": "0148", "type": "percent", "rate": 0.09, "source": "0013"},
        "Feriepenge": {"code": "0202", "type": "percent", "rate": 0.007, "source": "0013"},
        "A-indkomst uden bidrag": {"code": "0014", "type": "manual"},
        "B-indkomst med bidrag": {"code": "0036", "type": "manual"},
        "Kørselsgodtgørelse": {"code": "0048", "type": "manual"},
        "Fradrag": {"code": "0069", "type": "manual"},
    }

    # Initialize all codes with 0
    for entry in tax_config.values():
        df_payroll[entry["code"]] = 0

    # Apply percent and fixed taxes
    for name, entry in tax_config.items():
        if entry["type"] == "percent":
            df_payroll[entry["code"]] = df_payroll[entry["source"]] * entry["rate"]
        elif entry["type"] == "fixed":
            df_payroll[entry["code"]] = entry["value"]

    # Randomized manual entries
    unique_ids = df_payroll['employee_id'].unique().tolist()
    months = df_payroll["month"].unique().tolist()

    # Fradrag (e.g. personfradrag) to 2 random people/months
    for id_, month_ in zip(random.sample(unique_ids, 2), random.sample(months, 2)):
        condition = (df_payroll['employee_id'] == id_) & (df_payroll['month'] == month_)
        df_payroll.loc[condition, tax_config["Fradrag"]["code"]] = 8000

    # A-indkomst uden bidrag to 1 random employee
    for id_ in random.sample(unique_ids, len(unique_ids) // 100):
        df_payroll.loc[df_payroll['employee_id'] == id_, tax_config["A-indkomst uden bidrag"]["code"]] = 2000

    # Kørselsgodtgørelse to 60 random employees
    for id_ in random.sample(unique_ids, len(unique_ids) // 10):
        df_payroll.loc[df_payroll['employee_id'] == id_, tax_config["Kørselsgodtgørelse"]["code"]] = np.random.randint(100, 400)

    # B-indkomst med bidrag to 5 random employees
    for id_ in random.sample(unique_ids, len(unique_ids) // 20):
        df_payroll.loc[df_payroll['employee_id'] == id_, tax_config["B-indkomst med bidrag"]["code"]] = 5201
    
    return df_payroll

def add_LineID_Pay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms wide payroll dataframe to long format with PERIODE, FIRMA, MEDARBEJDER, Line_ID, BELØB.
    """
    line_id_cols = [
        '0013', '0014', '0015', '0016', '0019', '0020',
        '0026', '0036', '0046', '0048', '0069', '0147',
        '0148', '0202'
    ]

    df_long = df.melt(
        id_vars=['month', 'employee_id', "role_name", "first_name", "last_name"],
        value_vars=line_id_cols,
        var_name='line_id',
        value_name='amount'
    )


    return df_long


def split_Pay(df_base: pd.DataFrame, df_mapping: pd.DataFrame,
              min_split: int = 2, max_split: int = 3) -> pd.DataFrame:
    """
    Splits Amount across multiple Accounts based on Line_ID -> PayTypeKey mapping.
    """
    df_base = df_base.copy()
    df_mapping = df_mapping.copy()

    # Ensure consistent string format for Line_ID
    df_base['line_id'] = df_base['line_id'].astype(str).str.zfill(4)
    df_mapping['line_id'] = df_mapping['line_id'].astype(str).str.zfill(4)

    # Map Line_ID to list of PayTypeKeys (as strings to keep formatting consistent)
    mapping_dict = df_mapping.groupby('line_id')['Account'].apply(list).to_dict()

    new_rows = []

    for _, row in df_base.iterrows():
        line_id = row['line_id']
        amount = row['amount']
        paytypes = mapping_dict.get(line_id, [])

        if not paytypes:
            continue

        n = random.randint(min_split, max_split)

        if len(paytypes) < n:
            selected = [random.choice(paytypes) for _ in range(n)]
        else:
            selected = random.sample(paytypes, n)

        # Generate split proportions + noise
        proportions = np.random.dirichlet(np.ones(n))
        raw = proportions * amount 

        for ptk, amt in zip(selected, raw):
            new_rows.append({
                'month': row['month'],
                'employee_id': row['employee_id'],
                'role_name': row['role_name'],
                'first_name': row['first_name'],
                'last_name': row['last_name'],
                'line_id': line_id,
                'Account': int(ptk), 
                'amount': np.round(amt, -2)
            })

    return pd.DataFrame(new_rows)

def map_line_names(df: pd.DataFrame) -> pd.DataFrame:
    tax_config = {
        "A-skat": {"code": "0015", "type": "percent", "rate": 0.30, "source": "0013"},
        "AM-bidrag": {"code": "0016", "type": "percent", "rate": 0.08, "source": "0013"},
        "Fri bil": {"code": "0019", "type": "percent", "rate": 0.06, "source": "0013"},
        "Fri telefon": {"code": "0020", "type": "fixed", "value": 275},
        "Sundhedsforsikring": {"code": "0026", "type": "fixed", "value": 44},
        "ATP-bidrag": {"code": "0046", "type": "fixed", "value": 297},
        "Pension medarbejder": {"code": "0147", "type": "percent", "rate": 0.03, "source": "0013"},
        "Pension arbejdsgiver": {"code": "0148", "type": "percent", "rate": 0.09, "source": "0013"},
        "Feriepenge": {"code": "0202", "type": "percent", "rate": 0.007, "source": "0013"},
        "A-indkomst uden bidrag": {"code": "0014", "type": "manual"},
        "B-indkomst med bidrag": {"code": "0036", "type": "manual"},
        "Kørselsgodtgørelse": {"code": "0048", "type": "manual"},
        "Fradrag": {"code": "0069", "type": "manual"},
    }

    # Add the base code not in the config
    code_to_name = {"0013": "Monthly-pay"}
    # Reverse mapping from code to name
    code_to_name.update({v["code"]: k for k, v in tax_config.items()})

    df["line_name"] = df["line_id"].map(code_to_name)

    cols = ['month', 'employee_id', 'role_name', 'first_name', 'last_name', 'amount', 'line_id', 'line_name']

    return df[cols]

def create_full_payroll(df_payroll: pd.DataFrame, df_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Splits the full payroll data into multiple accounts based on Line_ID mapping.
    """
    df_payroll = add_LineID_Pay(df_payroll)
    #df_payroll = split_Pay(df_payroll, df_mapping)
    df_payroll = map_line_names(df_payroll)

    df_payroll = df_payroll.rename(columns={
        "employee_id": "employee_id",
        "role_name": "role_name",
        "first_name": "first_name",
        "last_name": "last_name",
        "line_id": "line_id",
        "line_name": "name",
        "amount": "amount",
        "month": "date"
    })
    
    cols_pay = ['date', 'employee_id', 'name', 'amount']
    df_payroll_keep = df_payroll[cols_pay]
    df_mapping = df_payroll[['name', 'line_id']]

    df_payroll_keep = df_payroll_keep.rename(columns={"name": "line_id"})
    return df_payroll_keep, df_mapping