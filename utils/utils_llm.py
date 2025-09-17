import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import openai

import dotenv
dotenv.load_dotenv()
API_KEY=dotenv.get_key('.env', 'api_key')

openai.api_key=API_KEY

def get_openai_client():
    return OpenAI(api_key=API_KEY)

def _target_account_types(name_column: str) -> List[str]:
    """
    Decide which account types are eligible for the item category.
    Matches your current logic one-to-one.
    """
    if name_column == "product_name":
        return ["Revenue"]
    elif name_column == "service_name":
        return ["Service Expense"]
    elif name_column == "procurement_name":
        return ["Product Expense"]
    else:
        raise ValueError(f"Unsupported name_column: {name_column}")


def _summarize_accounts_for_prompt(df_accounts: pd.DataFrame, allowed_types: List[str]) -> List[Dict[str, Any]]:
    """
    Keep the prompt compact but informative: id, name, type.
    """
    subset = df_accounts[df_accounts["account_type"].isin(allowed_types)].copy()
    cols = ["account_id", "name", "account_type"]
    subset = subset[cols].dropna()
    # cast account_id to str for LLM friendliness, but we will coerce back
    subset["account_id"] = subset["account_id"].astype(str)
    return subset.to_dict(orient="records")
# --- REPLACE your existing _llm_choose_accounts with this version ---

def _llm_choose_accounts(
    item_name: str,
    allowed_accounts: List[Dict[str, Any]],
    min_accounts: int,
    max_accounts: int,
    name_column: str,
    *,
    model: str = "gpt-4.1",
    temp: float = 0.2,
) -> Optional[List[Dict[str, Any]]]:
    """
    Uses OpenAI (via prompt_utils.get_openai_client) to select relevant GL accounts.
    Returns a list of dicts: [{"account_id": "string", "reason": "string"}, ...] or None on failure.
    """
    try:
        # local import to match your project structure
        import prompt_utils  # must expose get_openai_client()
    except Exception:
        return None

    # Bound #accounts similarly to your current behavior (2–6 by default)
    n_suggest = int(np.clip(np.random.randint(min_accounts, max_accounts + 1), min_accounts, max_accounts))

    # Keep the account list compact but unambiguous for the LLM
    # (account_id as string to avoid type confusion in the model)
    slim_accounts = []
    for a in allowed_accounts:
        slim_accounts.append({
            "account_id": str(a.get("account_id", "")),
            "name": str(a.get("name", "")),
            "account_type": str(a.get("account_type", "")),
        })

    system_msg = (
        "You are a meticulous financial mapping assistant. "
        "Given an item name and a list of available GL accounts, pick the most relevant accounts for that item. "
        "Output STRICT JSON ONLY with schema: "
        "{\"accounts\": [{\"account_id\": \"string\", \"reason\": \"string\"}, ...]} "
        "No extra keys, no commentary, no markdown."
    )

    user_payload = {
        "task": "Select GL accounts for an item.",
        "item_category": name_column,          # 'product_name' | 'service_name' | 'procurement_name'
        "item_name": item_name,
        "n_accounts": n_suggest,
        "available_accounts": slim_accounts,   # the pool you're allowed to choose from
        "selection_guidance": [
            "Use only account_ids present in available_accounts.",
            "Avoid duplicates; prefer diverse but relevant picks.",
            "Semantic alignment: products->Revenue, services->Service Expense, procurement->Product Expense.",
        ],
        "output_schema": {"accounts": [{"account_id": "string", "reason": "string"}]},
    }

    try:
        client = prompt_utils.get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=temp,
        )
        content = response.choices[0].message.content
        data = json.loads(content)

        if not isinstance(data, dict) or "accounts" not in data or not isinstance(data["accounts"], list):
            return None

        allowed_ids = {a["account_id"] for a in slim_accounts}
        cleaned = []
        seen = set()
        for entry in data["accounts"]:
            if not isinstance(entry, dict):
                continue
            aid = str(entry.get("account_id", "")).strip()
            if aid in allowed_ids and aid not in seen:
                seen.add(aid)
                cleaned.append({
                    "account_id": aid,
                    "reason": str(entry.get("reason", "")).strip()
                })

        return cleaned or None
    except Exception:
        return None


def _fallback_sample_accounts(df_accounts: pd.DataFrame, allowed_types: List[str], k: int) -> pd.DataFrame:
    pool = df_accounts[df_accounts["account_type"].isin(allowed_types)]
    if len(pool) == 0:
        raise ValueError(f"No accounts available for types {allowed_types}")
    k = min(k, max(1, len(pool)))
    return pool.sample(k, replace=(k > len(pool)))

def create_mapping_from_metadata(
    df_items: pd.DataFrame,
    df_accounts: pd.DataFrame,
    df_customers: Optional[pd.DataFrame] = None,
    df_vendors: Optional[pd.DataFrame] = None,
    *,
    name_column: str = "product_name",
    model: str = "gpt-4.1",
    client=None,
    min_accounts_per_item: int = 2,
    max_accounts_per_item: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    LLM-powered mapper for items -> accounts (with optional customers/vendors).
    
    Inputs:
      - df_items: dataframe with at least a 'name' column (the item name)
      - df_accounts: dataframe with ['account_id','name','account_type']
      - df_customers: dataframe with customer names (optional)
      - df_vendors: dataframe with vendor names (optional)

    Behavior:
      - name_column == 'product_name'     -> eligible account_type: 'Revenue'
      - name_column == 'service_name'     -> eligible account_type: 'Service Expense'
      - name_column == 'procurement_name' -> eligible account_type: 'Product Expense'

    Output:
      One row per (item, account_id), with:
        - name
        - account_id
        - account_name
        - customer_name   (for product_name if customers given)
        - vendor_name     (for service_name/procurement_name if vendors given)
    """
    rng = np.random.RandomState(seed)

    # Defensive copies and dtypes
    df_items = df_items.copy()
    df_accounts = df_accounts.copy()

    if "name" not in df_items.columns:
        raise ValueError("df_items must contain a 'name' column for item names.")
    required_acc_cols = {"account_id", "name", "account_type"}
    if not required_acc_cols.issubset(set(df_accounts.columns)):
        raise ValueError(f"df_accounts must contain columns: {sorted(required_acc_cols)}")

    # Decide allowed account types
    allowed_types = _target_account_types(name_column)
    allowed_accounts_prompt = _summarize_accounts_for_prompt(df_accounts, allowed_types)

    # Prepare client
    client = _get_client(client)

    mappings: List[Dict[str, Any]] = []

    for _, row in df_items.iterrows():
        item_value = str(row["name"]).strip()

        # Ask LLM to select accounts
        llm_accounts = _llm_choose_accounts(
            client=client,
            model=model,
            item_name=item_value,
            allowed_accounts=allowed_accounts_prompt,
            min_accounts=min_accounts_per_item,
            max_accounts=max_accounts_per_item,
            name_column=name_column,
        )

        if llm_accounts:
            picked_ids = [a["account_id"] for a in llm_accounts]
            acc_tmp = df_accounts.copy()
            acc_tmp["account_id_str"] = acc_tmp["account_id"].astype(str)
            chosen = acc_tmp[acc_tmp["account_id_str"].isin(picked_ids)]
            if chosen.empty:
                chosen = _fallback_sample_accounts(df_accounts, allowed_types, rng.randint(min_accounts_per_item, max_accounts_per_item + 1))
        else:
            chosen = _fallback_sample_accounts(df_accounts, allowed_types, rng.randint(min_accounts_per_item, max_accounts_per_item + 1))

        for _, acc in chosen.iterrows():
            mapping = {
                "name": item_value,
                "account_id": acc["account_id"],
                "account_name": acc["name"],
            }

            if name_column == "product_name" and isinstance(df_customers, pd.DataFrame) and not df_customers.empty:
                customer = df_customers.sample(1, random_state=rng).iloc[0]
                mapping["customer_name"] = customer.get("name", "")

            if name_column in ("service_name", "procurement_name") and isinstance(df_vendors, pd.DataFrame) and not df_vendors.empty:
                vendor = df_vendors.sample(1, random_state=rng).iloc[0]
                mapping["vendor_name"] = vendor.get("name", "")

            mappings.append(mapping)

    out = pd.DataFrame(mappings)

    if "customer_name" not in out.columns:
        out["customer_name"] = np.nan
    if "vendor_name" not in out.columns:
        out["vendor_name"] = np.nan

    col_order = ["name", "account_id", "account_name", "customer_name", "vendor_name"]
    out = out[[c for c in col_order if c in out.columns]]

    return out
