import pandas as pd
from generators.random_generators import generate_dim_date, generate_document_metadata
import random
import numpy as np

def assign_volume_and_value(df_products: pd.DataFrame, non_linear_multiplier: int) -> pd.DataFrame:
    """
    Add realistic simulated volume and spend data to a GPT-generated product list.

    Enhancements:
    - Expensive products → lower volume
    - Cheap products → higher volume
    - Higher volumes especially for cheap products towards bottom of the list
    - Scales with DataFrame length

    Returns:
        DataFrame with:
            - ProductName
            - UnitPriceDKK
            - SimulatedVolume
            - TotalSpendDKK
    """
    
    df = df_products.copy()

    if "UnitPriceDKK" not in df.columns:
        raise ValueError("Missing 'UnitPriceDKK' in input DataFrame")

    np.random.seed(42)
    df["UnitPriceDKK"] = pd.to_numeric(df["UnitPriceDKK"], errors="coerce").fillna(0)

    # Normalize prices
    price_norm = df["UnitPriceDKK"] / (df["UnitPriceDKK"].max() + 1e-6)
    inverse_price_weight = 1 - price_norm  # Cheaper → higher weight

    # Rank to prioritize bottom of list
    df["IndexRankNorm"] = df.reset_index().index / len(df)

    # Noise and inflation towards bottom for cheap products
    noise = np.random.lognormal(mean=0.0, sigma=0.4, size=len(df))

    # The rank multiplier boosts volume for cheaper products towards the bottom
    # Nonlinear boost: higher rank → exponentially higher volume
    # This ensures that cheaper products at the bottom get a significant volume boost
    rank_multiplier = 1 + (df["IndexRankNorm"] ** 2) * non_linear_multiplier  # nonlinear boost

    # Adjusted volume generation
    base_volume = inverse_price_weight * noise * rank_multiplier * 100

    # Clip to realistic values
    volume_clipped = np.clip(np.round(base_volume), 1, 2000)

    df["SimulatedVolume"] = volume_clipped.astype(int)
    df["TotalSpendDKK"] = np.round(df["SimulatedVolume"] * df["UnitPriceDKK"], 2)

    # Optional cleanup
    df.drop(columns=["IndexRankNorm"], inplace=True)

    return df

def assign_account_ids(account_types: pd.Series) -> pd.Series:
    """
    Generate 5-digit account numbers based on account type prefix:
    - Asset: starts with 1
    - Liability: starts with 2
    - Equity: starts with 3
    - Revenue: starts with 4
    - Expense: starts with 5
    """
    prefix_map = {
        'Asset': '1',
        'Liability': '2',
        'Equity': '3',
        'Revenue': '4',
        'Expense': '5'
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

def split_fact_into_gl(df_fact: pd.DataFrame, df_accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Naive splitter: for each fact entry, randomly pick 2–3 accounts and generate 
    debit/credit entries summing to 0.
    """
    rows = []

    for _, row in df_fact.iterrows():
        amount = row["AmountLocal"]
        accounts = df_accounts.sample(2)
        split1 = round(amount * random.uniform(0.4, 0.6), 2)
        split2 = amount - split1

        rows.append({**row, "Account_ID": accounts.iloc[0]["Account_ID"], "Amount": split1})
        rows.append({**row, "Account_ID": accounts.iloc[1]["Account_ID"], "Amount": -split1})

    return pd.DataFrame(rows)


def estimate_costs_from_payroll(
    df_pay: pd.DataFrame,
    product_multiplier: float = 3,
    service_multiplier: float = 1.5,
    overhead_multiplier: float = 0.2
) -> dict:
    """
    Estimate total product, service, and overhead costs based on payroll data.

    Args:
        df_pay (pd.DataFrame): Monthly payroll entries. Must include 'BELØB'.
        product_multiplier (float): Factor to estimate product cost.
        service_multiplier (float): Factor to estimate service cost.
        overhead_multiplier (float): Factor to estimate overhead.

    Returns:
        dict: Estimated cost breakdown.
    """
    total_payroll = df_pay["MonthlyPay"].sum()

    return {
        "EstimatedPayroll": total_payroll,
        "EstimatedProductCost": round(total_payroll * product_multiplier),
        "EstimatedServiceCost": round(total_payroll * service_multiplier),
        "EstimatedOverhead": round(total_payroll * overhead_multiplier)
    }

def create_erp_data(
        df_expenses: pd.DataFrame,
        df_expenses_mapping: pd.DataFrame,
        df_document_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create ERP journal lines from spend data and mapping.

        Args:
            df_expenses: Spend data with 'ItemName', 'TotalAnnualSpend', 'SourceType'
            df_expenses_mapping: Mapping with 'GLAccount', 'CostCenter', optionally 'Department', 'CustomerName'
            df_document_metadata: Pre-generated document metadata with 'DocumentNumber', 'Currency'
            df_date: Pre-generated list of dates

        Returns:
            ERP journal DataFrame
        """
        df_expenses = assign_split_count(df_expenses)
        df_date = generate_dim_date()

        df = pre_split_spend_lines(
            df_spend=df_expenses,
            df_mapping=df_expenses_mapping,
            n_lines_per_item=30
        )

        df = assign_to_documents(
            df_lines=df,
            df_documentnumber=df_document_metadata,
            df_date=df_date.sample(1000, replace=True)
        )

        if "CustomerName" not in df.columns:
            df = patch_unbalanced_documents(df, tolerance=10)

        exchange_rate = 7.45
        df["AmountEUR"] = (df["Amount"] / exchange_rate).round(2)
        df.rename(columns={"Amount": "AmountDKK"}, inplace=True)

        cols = ['DocumentNumber', 'Date', 'Currency', 'AmountDKK', 'AmountEUR', 'Type', 'GLAccount', 'CostCenter']
        if "Department" in df.columns:
            cols.append("Department")
        if "CustomerName" in df.columns:
            cols.append("CustomerName")
        cols += ["ItemName", "SourceType"]

        return df[cols]

def pre_split_spend_lines(
    df_spend,
    df_mapping,
    n_lines_per_item=30
):
    """
    Splits each item (product or service) into many credit AND debit lines,
    sampled independently from the mapping.

    Assumes:
        - df_spend includes 'ItemName', 'TotalAnnualSpend', and 'SourceType'
        - df_mapping includes 'ItemName' and relevant account info

    Returns:
        DataFrame with one row per ERP journal line
    """
    np.random.seed(42)
    line_items = []

    for _, row in df_spend.iterrows():
        item_name = row["ItemName"]
        source_type = row["SourceType"]
        total = row["TotalAnnualSpend"]

        weights = np.random.dirichlet(np.ones(n_lines_per_item) * 0.5)
        credit_amounts = weights * total

        credit_mappings = df_mapping[df_mapping["ItemName"] == item_name].sample(n=n_lines_per_item, replace=True)
        debit_mappings = df_mapping[df_mapping["ItemName"] == item_name].sample(n=n_lines_per_item, replace=True)

        for amt, (_, map_row) in zip(credit_amounts, credit_mappings.iterrows()):
            entry = {
                "Type": "Credit",
                "Amount": round(-amt, 2),
                "GLAccount": map_row["GLAccount"],
                "CostCenter": map_row.get("CostCenter"),
                "ItemName": item_name,
                "SourceType": source_type
            }
            if "Department" in map_row:
                entry["Department"] = map_row["Department"]
            if "CustomerName" in map_row:
                entry["CustomerName"] = map_row["CustomerName"]
            line_items.append(entry)

        for amt, (_, map_row) in zip(credit_amounts, debit_mappings.iterrows()):
            entry = {
                "Type": "Debit",
                "Amount": round(amt, 2),
                "GLAccount": map_row["GLAccount"],
                "CostCenter": map_row.get("CostCenter"),
                "ItemName": item_name,
                "SourceType": source_type
            }
            if "Department" in map_row:
                entry["Department"] = map_row["Department"]
            if "CustomerName" in map_row:
                entry["CustomerName"] = map_row["CustomerName"]
            line_items.append(entry)

    return pd.DataFrame(line_items)


def assign_split_count(df_spend: pd.DataFrame, non_linear_multiplier: int = 10,
                       min_splits: int = 2, max_splits: int = 20) -> pd.DataFrame:
    """
    Assigns 'NumSplits' to each product based on proportionality of spend.
    Smaller proportionality → more splits (e.g. small frequent items).
    
    Args:
        df_spend (pd.DataFrame): Must include 'Proportionality' column (0–1 range).
        non_linear_multiplier (int): Boosts splits for low-ranked rows.
        min_splits (int): Minimum number of splits per product.
        max_splits (int): Maximum number of splits per product.

    Returns:
        pd.DataFrame: With new 'NumSplits' column.
    """
    df = df_spend.copy()

    if "Proportionality" not in df.columns:
        raise ValueError("Missing 'Proportionality' column in df_spend")

    np.random.seed(42)

    # Invert proportionality: higher = fewer splits
    inv_weight = 1 - df["Proportionality"]
    inv_weight = inv_weight / (inv_weight.max() + 1e-6)

    # Rank boost for bottom-of-list items
    df["IndexRankNorm"] = df.reset_index().index / len(df)
    rank_multiplier = 1 + (df["IndexRankNorm"] ** 2) * non_linear_multiplier

    # Log-normal noise
    noise = np.random.lognormal(mean=0.0, sigma=0.4, size=len(df))

    raw_splits = inv_weight * noise * rank_multiplier * 3

    # Normalize to min/max split range
    normalized = (raw_splits - raw_splits.min()) / (raw_splits.max() - raw_splits.min())
    df["NumSplits"] = (min_splits + normalized * (max_splits - min_splits)).round().astype(int)

    df.drop(columns=["IndexRankNorm"], inplace=True)
    return df

def assign_to_documents(df_lines, df_documentnumber, df_date):
    """
    Assigns journal lines to documents, ensuring:
    - Each document contains only one currency
    - Each document balances to zero (adds correction if needed)
    - Random posting dates

    Returns:
        Finalized ERP journal as DataFrame
    """
    df = df_lines.copy()
    final_rows = []

    # Group doc numbers by currency
    docs_by_currency = (
        df_documentnumber.groupby("Currency")["DocumentNumber"]
        .apply(list)
        .to_dict()
    )

    # Assign currency randomly per line, then sample document and date
    df["Currency"] = np.random.choice(list(docs_by_currency.keys()), size=len(df))

    # Assign document and date per row based on currency
    doc_mapping = {
        currency: list(np.random.choice(docs, size=(df["Currency"] == currency).sum(), replace=True))
        for currency, docs in docs_by_currency.items()
    }

    df["DocumentNumber"] = df["Currency"].map(lambda c: doc_mapping[c].pop())
    df["Date"] = np.random.choice(df_date["Date"], size=len(df))

    # Ensure balance per document (adds 1 correction row if needed)
    for doc_id, group in df.groupby("DocumentNumber"):
        group = group.copy()
        imbalance = round(group["Amount"].sum(), 2)

        if abs(imbalance) > 0.01:
            correction_type = "Debit" if imbalance < 0 else "Credit"
            correction_amount = round(abs(imbalance), 2)

            sample_row = group.sample(1).iloc[0].copy()
            correction_row = {
                "DocumentNumber": doc_id,
                "Date": sample_row["Date"],
                "Currency": sample_row["Currency"],
                "Amount": correction_amount if correction_type == "Debit" else -correction_amount,
                "Type": correction_type,
                "GLAccount": sample_row["GLAccount"],
                "CostCenter": sample_row["CostCenter"],
                "ItemName": sample_row["ItemName"],
                "SourceType": sample_row.get("SourceType", "Correction")
            }
            if "Department" in sample_row:
                correction_row["Department"] = sample_row["Department"]
            if "CustomerName" in sample_row:
                correction_row["CustomerName"] = sample_row["CustomerName"]

            final_rows.append(correction_row)

        final_rows.extend(group.to_dict("records"))

    return pd.DataFrame(final_rows)

def patch_unbalanced_documents(df: pd.DataFrame, tolerance: float = 10) -> pd.DataFrame:
    """
    Adds a balancing debit or credit line for each DocumentNumber that doesn't sum to 0.
    Metadata (GLAccount, Department, CostCenter, etc.) is sampled from existing lines in the document.
    """
    df = df.copy()
    patched_rows = []
    grouped = df.groupby("DocumentNumber")

    for doc_id, group in grouped:
        imbalance = round(group["Amount"].sum(), 2)
        if abs(imbalance) > tolerance:
            correction_type = "Debit" if imbalance < 0 else "Credit"
            correction_amount = round(abs(imbalance), 2)
            sample_row = group.sample(1).iloc[0]

            correction_row = {
                "DocumentNumber": doc_id,
                "Date": sample_row["Date"],
                "Currency": sample_row["Currency"],
                "Amount": correction_amount if correction_type == "Debit" else -correction_amount,
                "Type": correction_type,
                "GLAccount": sample_row["GLAccount"],
                "CostCenter": sample_row["CostCenter"],
                "ItemName": sample_row.get("ItemName", "Correction Line"),
                "SourceType": sample_row.get("SourceType", "Correction")
            }

            if "Department" in sample_row:
                correction_row["Department"] = sample_row["Department"]
            if "CustomerName" in sample_row:
                correction_row["CustomerName"] = sample_row["CustomerName"]

            patched_rows.append(correction_row)

    if patched_rows:
        df = pd.concat([df, pd.DataFrame(patched_rows)], ignore_index=True)

    return df
