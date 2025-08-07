import pandas as pd
import numpy as np
from generators.random_generators import generate_dim_date, generate_document_metadata


def estimate_costs_from_payroll(
    df_pay: pd.DataFrame,
    product_multiplier: float = 3,
    service_multiplier: float = 1.5,
    overhead_multiplier: float = 0.2,
    revenue_multiplier: float = 7,
) -> dict:
    """
    Estimate total product, service, and overhead costs based on payroll data.
    """
    temp_df = df_pay[df_pay["line_id"] == "Monthly-pay"]
    total_payroll = temp_df["amount"].sum()

    return {
        "estimated_payroll": total_payroll,
        "estimated_product": round(total_payroll * product_multiplier),
        "estimated_service": round(total_payroll * service_multiplier),
        "estimated_overhead": round(total_payroll * overhead_multiplier),
        "estimated_revenue": round(total_payroll * revenue_multiplier)
    }


def assign_split_count(
    df_spend: pd.DataFrame,
    non_linear_multiplier: int = 10,
    min_splits: int = 2,
    max_splits: int = 20
) -> pd.DataFrame:
    """
    Assigns 'num_splits' to each product based on proportionality of spend.
    Smaller proportionality → more splits.
    """
    df = df_spend.copy()

    if "proportionality" not in df.columns:
        raise ValueError("Missing 'proportionality' column in df_spend")

    np.random.seed(42)

    inv_weight = 1 - df["proportionality"]
    inv_weight = inv_weight / (inv_weight.max() + 1e-6)

    df["index_rank"] = df.reset_index().index / len(df)
    rank_multiplier = 1 + (df["index_rank"] ** 2) * non_linear_multiplier

    noise = np.random.lognormal(mean=0.0, sigma=0.4, size=len(df))
    raw_splits = inv_weight * noise * rank_multiplier * 3

    normalized = (raw_splits - raw_splits.min()) / (raw_splits.max() - raw_splits.min())
    df["num_splits"] = (min_splits + normalized * (max_splits - min_splits)).round().astype(int)

    df.drop(columns=["index_rank"], inplace=True)
    return df


def pre_split_spend_lines(
    df_spend: pd.DataFrame,
    df_mapping: pd.DataFrame,
    n_lines_per_item: int = 30
) -> pd.DataFrame:
    """
    Splits each item (product or service) into journal lines,
    with expenses as negative and revenues as positive values.
    """
    line_items = []

    for _, row in df_spend.iterrows():
        item_name = row["item_name"]
        source_type = row["source_type"]
        total = row["annual_spend"]

        product_id = row.get("product_id")
        procurement_id = row.get("procurement_id")
        service_id = row.get("service_id")

        weights = np.random.dirichlet(np.ones(n_lines_per_item) * 0.5)
        amounts = weights * total

        mappings = df_mapping[df_mapping["item_name"] == item_name].sample(n=n_lines_per_item, replace=True)

        for amt, (_, map_row) in zip(amounts, mappings.iterrows()):
            if source_type.lower() in ["service", "procurement", "overhead"]:
                signed_amount = -round(amt, 2)
            elif source_type.lower() == "product":
                signed_amount = round(amt, 2)
            else:
                signed_amount = round(amt, 2)

            entry = {
                "type": "Entry",
                "amount": signed_amount,
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
            if "vendor_name" in map_row:
                entry["vendor_name"] = map_row["vendor_name"]

            line_items.append(entry)

    return pd.DataFrame(line_items)


def create_erp_data(
    df_expenses: pd.DataFrame,
    df_expenses_mapping: pd.DataFrame,
    df_document_metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Create ERP journal lines from spend data and mapping.
    Expenses are negative, revenues are positive. No balancing logic.
    """
    df_expenses = assign_split_count(df_expenses)
    df_date = generate_dim_date(year_start=2020, year_end=2025)

    df = pre_split_spend_lines(
        df_spend=df_expenses,
        df_mapping=df_expenses_mapping,
        n_lines_per_item=30
    )

    # Assign document number, date, and currency randomly (no balancing)
    df["currency"] = np.random.choice(df_document_metadata["currency"], size=len(df), replace=True)
    df["document_number"] = np.random.choice(df_document_metadata["document_number"], size=len(df), replace=True)
    df["date"] = np.random.choice(df_date["date"], size=len(df), replace=True)

    # Currency conversion
    exchange_rate = 7.45
    df["amount_eur"] = (df["amount"] / exchange_rate).round(2)
    df.rename(columns={"amount": "amount_dkk"}, inplace=True)

    # Final output columns
    cols = ['document_number', 'date', 'currency', 'amount_dkk', 'amount_eur', 'type',
            'account_id', 'account_name', 'product_id', 'procurement_id', 'service_id']

    if "department_name" in df.columns:
        cols.append("department_name")
    if "customer_name" in df.columns:
        cols.append("customer_name")
    if "vendor_name" in df.columns:
        cols.append("vendor_name")

    return df[cols]

def balance_documents_with_assets(
    df_erp: pd.DataFrame,
    df_accounts: pd.DataFrame,
    tolerance: float = 100
) -> pd.DataFrame:
    """
    Balances each document_number so its amount sums to 0,
    using exactly 3 'Debit' lines from accounts with account_type == 'Asset'.

    Args:
        df_erp: ERP journal data
        df_accounts: Dimension table with 'account_id', 'account_type'
        tolerance: Allowable imbalance without correction

    Returns:
        Balanced ERP DataFrame
    """
    df = df_erp.copy()
    asset_accounts = df_accounts[df_accounts["account_type"] == "Asset"]

    correction_rows = []

    for doc_id, group in df.groupby("document_number"):
        imbalance = round(group["amount"].sum(), 2)

        if abs(imbalance) > tolerance:
            sample_row = group.sample(1).iloc[0]
            correction_sign = -1 if imbalance > 0 else 1
            total_correction = abs(imbalance)

            # Split into exactly 3 chunks
            weights = np.random.dirichlet([1, 1, 1])
            chunk_values = np.round(weights * total_correction, 2)

            # Adjust the final value to ensure exact balancing
            chunk_values[-1] = round(total_correction - chunk_values[:-1].sum(), 2)

            for amt in chunk_values:
                asset_row = asset_accounts.sample(1).iloc[0]
                signed_amt = correction_sign * amt

                correction_entry = {
                    "document_number": doc_id,
                    "debit_credit": "Debit",
                    "date": sample_row["date"],
                    "amount": signed_amt,
                    "account_id": asset_row["name"],
                    "product_id": None,
                    "procurement_id": None,
                    "service_id": None,
                }

                if "department_name" in sample_row:
                    correction_entry["department_name"] = sample_row["department_name"]
                if "customer_name" in sample_row:
                    correction_entry["customer_name"] = sample_row["customer_name"]
                if "vendor_name" in sample_row:
                    correction_entry["vendor_name"] = sample_row["vendor_name"]

                correction_rows.append(correction_entry)

    if correction_rows:
        df = pd.concat([df, pd.DataFrame(correction_rows)], ignore_index=True)

    return df

