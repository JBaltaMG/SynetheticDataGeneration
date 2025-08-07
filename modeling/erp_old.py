import pandas as pd
from generators.random_generators import generate_dim_date, generate_document_metadata
import random
import numpy as np

def estimate_costs_from_payroll(
    df_pay: pd.DataFrame,
    product_multiplier: float = 3,
    service_multiplier: float = 1.5,
    overhead_multiplier: float = 0.2,
    revenue_multiplier: float = 7,
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
    temp_df = df_pay[df_pay["line_id"]=="Monthly-pay"]
    total_payroll = temp_df["amount"].sum()

    return {
        "estimated_payroll": total_payroll,
        "estimated_product": round(total_payroll * product_multiplier),
        "estimated_service": round(total_payroll * service_multiplier),
        "estimated_overhead": round(total_payroll * overhead_multiplier),
        "estimated_revenue": round(total_payroll * revenue_multiplier)
    }

def create_erp_data(
    df_expenses: pd.DataFrame,
    df_expenses_mapping: pd.DataFrame,
    df_document_metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Create ERP journal lines from spend data and mapping.

    Args:
        df_expenses: Spend data with 'item_name', 'annual_spend', 'source_type'
        df_expenses_mapping: Mapping with 'account_id', 'CostCenter', optionally 'department_name', 'customer_name'
        df_document_metadata: Pre-generated document metadata with 'document_number', 'currency'

    Returns:
        ERP journal DataFrame with product_id, procurement_id, and service_id columns
    """

    df_expenses = assign_split_count(df_expenses)
    df_date = generate_dim_date(year_start=2020, year_end=2025)

    df = pre_split_spend_lines(
        df_spend=df_expenses,
        df_mapping=df_expenses_mapping,
        n_lines_per_item=30
    )

    df = assign_to_documents(
        df_lines=df,
        df_document_number=df_document_metadata,
        df_date=df_date.sample(1000, replace=True)
    )

    if "customer_name" not in df.columns:
        df = patch_unbalanced_documents(df, tolerance=10)

    # currency conversion
    exchange_rate = 7.45
    df["amount_eur"] = (df["amount"] / exchange_rate).round(2)
    df.rename(columns={"amount": "amount_dkk"}, inplace=True)

    # Build final column list
    cols = ['document_number', 'date', 'currency', 'amount_dkk', 'amount_eur', 'type', 'account_id', 'account_name', "product_id", "procurement_id", "service_id"]
    if "department_name" in df.columns:
        cols.append("department_name")
    if "customer_name" in df.columns:
        cols.append("customer_name")

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
        - df_spend includes 'item_name', 'annual_spend', 'source_type'
        - df_mapping includes 'item_name' and relevant account info

    Returns:
        DataFrame with one row per ERP journal line
    """

    line_items = []

    for _, row in df_spend.iterrows():
        item_name = row["item_name"]
        source_type = row["source_type"]
        total = row["annual_spend"]

        # Track ids
        product_id = row.get("product_id")
        procurement_id = row.get("procurement_id")
        service_id = row.get("service_id")

        weights = np.random.dirichlet(np.ones(n_lines_per_item) * 0.5)
        credit_amounts = weights * total

        credit_mappings = df_mapping[df_mapping["item_name"] == item_name].sample(n=n_lines_per_item, replace=True)
        debit_mappings = df_mapping[df_mapping["item_name"] == item_name].sample(n=n_lines_per_item, replace=True)

        for amt, (_, map_row) in zip(credit_amounts, credit_mappings.iterrows()):
            entry = {
                "type": "Credit",
                "amount": round(-amt, 2),
                "account_name": map_row["account_name"],
                "account_id": map_row["account_id"],
                "item_name": item_name,
                "source_type": source_type,
                "product_id": product_id,
                "procurement_id": procurement_id,
                "service_id": service_id
            }
            if "department_name" in map_row:
                entry["department_name"] = map_row["department_name"]
            if "customer_name" in map_row:
                entry["customer_name"] = map_row["customer_name"]
            line_items.append(entry)

        for amt, (_, map_row) in zip(credit_amounts, debit_mappings.iterrows()):
            entry = {
                "type": "Debit",
                "amount": round(amt, 2),
                "account_id": map_row["account_id"],
                "account_name": map_row["account_name"],
                "item_name": item_name,
                "source_type": source_type,
                "product_id": product_id,
                "procurement_id": procurement_id,
                "service_id": service_id
            }
            if "department_name" in map_row:
                entry["department_name"] = map_row["department_name"]
            if "customer_name" in map_row:
                entry["customer_name"] = map_row["customer_name"]
            line_items.append(entry)

    return pd.DataFrame(line_items)



def assign_split_count(df_spend: pd.DataFrame, non_linear_multiplier: int = 10,
                       min_splits: int = 2, max_splits: int = 20) -> pd.DataFrame:
    """
    Assigns 'NumSplits' to each product based on proportionality of spend.
    Smaller proportionality → more splits (e.g. small frequent items).
    
    Args:
        df_spend (pd.DataFrame): Must include 'proportionality' column (0–1 range).
        non_linear_multiplier (int): Boosts splits for low-ranked rows.
        min_splits (int): Minimum number of splits per product.
        max_splits (int): Maximum number of splits per product.

    Returns:
        pd.DataFrame: With new 'NumSplits' column.
    """
    df = df_spend.copy()

    if "proportionality" not in df.columns:
        raise ValueError("Missing 'proportionality' column in df_spend")

    np.random.seed(42)

    # Invert proportionality: higher = fewer splits
    inv_weight = 1 - df["proportionality"]
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
def assign_to_documents(df_lines, df_document_number, df_date):
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
        df_document_number.groupby("currency")["document_number"]
        .apply(list)
        .to_dict()
    )

    # Assign currency randomly per line, then sample document and date
    df["currency"] = np.random.choice(list(docs_by_currency.keys()), size=len(df))

    # Assign document and date per row based on currency
    doc_mapping = {
        currency: list(np.random.choice(docs, size=(df["currency"] == currency).sum(), replace=True))
        for currency, docs in docs_by_currency.items()
    }

    df["document_number"] = df["currency"].map(lambda c: doc_mapping[c].pop())
    df["date"] = np.random.choice(df_date["date"], size=len(df))

    # Ensure balance per document (adds 1 correction row if needed)
    for doc_id, group in df.groupby("document_number"):
        group = group.copy()
        imbalance = round(group["amount"].sum(), 2)

        if abs(imbalance) > 0.01:
            correction_type = "Debit" if imbalance < 0 else "Credit"
            correction_amount = round(abs(imbalance), 2)

            sample_row = group.sample(1).iloc[0].copy()
            correction_row = {
                "document_number": doc_id,
                "date": sample_row["date"],
                "currency": sample_row["currency"],
                "amount": correction_amount if correction_type == "Debit" else -correction_amount,
                "type": correction_type,
                "account_id": sample_row["account_id"],
                "account_name": sample_row["account_name"],
                "item_name": sample_row["item_name"],
                "source_type": sample_row.get("source_type", "Correction"),
                "product_id": sample_row.get("product_id"),
                "procurement_id": sample_row.get("procurement_id"),
                "service_id": sample_row.get("service_id")
            }
            if "department_name" in sample_row:
                correction_row["department_name"] = sample_row["department_name"]
            if "customer_name" in sample_row:
                correction_row["customer_name"] = sample_row["customer_name"]

            final_rows.append(correction_row)

        final_rows.extend(group.to_dict("records"))

    return pd.DataFrame(final_rows)

def patch_unbalanced_documents(df: pd.DataFrame, tolerance: float = 10) -> pd.DataFrame:
    """
    Adds multiple small balancing debit/credit lines for each document_number that doesn't sum to 0.
    Metadata (account_id, Department, CostCenter, etc.) is sampled from existing lines in the document.
    """
    df = df.copy()
    patched_rows = []
    grouped = df.groupby("document_number")

    for doc_id, group in grouped:
        imbalance = round(group["amount"].sum(), 2)

        if abs(imbalance) > tolerance:
            correction_type = "Debit" if imbalance < 0 else "Credit"
            correction_sign = 1 if correction_type == "Debit" else -1
            remaining = abs(imbalance)

            # Break into small chunks
            np.random.seed(42)
            chunks = []
            while remaining > 0:
                step = round(np.random.uniform(0, 10000), 2)
                if step > remaining:
                    step = round(remaining, 2)
                chunks.append(step)
                remaining -= step

            sample_row = group.sample(1).iloc[0]

            for amt in chunks:
                correction_row = {
                    "document_number": doc_id,
                    "date": sample_row["date"],
                    "currency": sample_row["currency"],
                    "amount": correction_sign * amt,
                    "type": correction_type,
                    "account_id": sample_row["account_id"],
                    "account_name": sample_row["account_name"],
                    "CostCenter": sample_row.get("CostCenter"),
                    "item_name": sample_row.get("item_name", "Correction Line"),
                    "source_type": sample_row.get("source_type", "Correction"),
                    "product_id": sample_row.get("product_id"),
                    "procurement_id": sample_row.get("procurement_id"),
                    "service_id": sample_row.get("service_id")
                }

                if "department_name" in sample_row:
                    correction_row["department_name"] = sample_row["department_name"]
                if "customer_name" in sample_row:
                    correction_row["customer_name"] = sample_row["customer_name"]

                patched_rows.append(correction_row)

    if patched_rows:
        df = pd.concat([df, pd.DataFrame(patched_rows)], ignore_index=True)

    return df
