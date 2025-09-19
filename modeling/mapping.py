
import utils.utils as utils
from concurrent.futures import ThreadPoolExecutor, as_completed
import re, hashlib, json, dotenv
from openai import OpenAI
import random
import numpy as np
import pandas as pd
import os
from utils.utils_llm import create_mapping_from_metadata
from typing import Dict, Any, List, Optional
import time
import random

def _overlap_tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))

def _shortlist_accounts(df_accounts, allowed, item_name, k=80):
    toks = _overlap_tokens(item_name)
    sub = df_accounts[df_accounts["GLLevel02"] == allowed][["AccountKey","name","GLLevel02"]].dropna().copy()
    sub["AccountKey"] = sub["AccountKey"].astype(str)
    sub["__score"] = sub["name"].astype(str).apply(lambda s: len(toks & _overlap_tokens(s)))
    sub = sub.sort_values("__score", ascending=False).head(k).drop(columns="__score")

    records = []
    for i, r in sub.reset_index(drop=True).iterrows():
        records.append({
            "idx": i,
            "AccountKey": r["AccountKey"],
            "name": str(r["name"]),             # ← use the column, not r.name
            "GLLevel02": r["GLLevel02"],
        })
    return records

def _shortlist_partners(df_partners, item_name, account_name, k=200):
    if df_partners is None or "name" not in df_partners.columns or df_partners.empty:
        return []
    toks = _overlap_tokens(item_name) | _overlap_tokens(account_name)
    sub = df_partners.copy()
    sub["__score"] = sub["name"].astype(str).apply(lambda s: len(toks & _overlap_tokens(s)))
    sub = sub.sort_values("__score", ascending=False).head(k).drop(columns="__score")
    return [{"idx":i,"name":n} for i,n in enumerate(sub["name"].astype(str).tolist())]

def _det_pick_idx(n: int, key: str) -> int:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h, 16) % max(1, n)

def _pick_idx_with_gpt(client, system_schema_text, payload, n, fallback_key, model="gpt-4.1", retries=2):
    backoff = 0.6
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content":system_schema_text},
                    {"role":"user","content":json.dumps(payload)},
                ],
                temperature=1,
            )
            data = json.loads(resp.choices[0].message.content)
            idx = next(iter(data.values()))
            if isinstance(idx, int) and 0 <= idx < n:
                return idx
        except Exception:
            pass
        if attempt < retries:
            time.sleep(backoff + random.random() * 0.3)
            backoff *= 1.6
    # deterministic fallback
    return _det_pick_idx(n, fallback_key)


def _parse_refs(val: object) -> set[str]:
    """Accept NaN or string; split on ; | , and collapse whitespace."""
    if pd.isna(val):
        return set()
    parts = re.split(r"[;|,]+", str(val))
    return {p.strip() for p in parts if p and p.strip()}


def create_mapping_parallel_simple(
    df_items: pd.DataFrame,
    df_accounts: pd.DataFrame,
    df_customers: Optional[pd.DataFrame]=None,
    df_vendors: Optional[pd.DataFrame]=None,
    *,
    df_bu_companies: Optional[pd.DataFrame]=None,   # expects bu_key, sells_to, buys_from
    name_column: str="product_name",
    model: str="gpt-4.1",
    max_workers: int=12
) -> pd.DataFrame:

    # --- validation (as before) ---
    for req in [["name"]]:
        miss = [c for c in req if c not in df_items.columns]
        if miss: raise ValueError(f"df_items missing {miss}")
    miss = [c for c in ["AccountKey","name","GLLevel02"] if c not in df_accounts.columns]
    if miss: raise ValueError(f"df_accounts missing {miss}")

    ACC_ALLOWED = {
        "product_name": "Net Sales",
        "service_name": "Operating Expenses",
        "procurement_name": "Cost Of Sales",
    }[name_column]

    dotenv.load_dotenv()
    api_key = dotenv.get_key(".env","api_key")
    client = OpenAI(api_key=api_key)

    # --- NEW: build BU relation maps ---
    bu_key_set: set[str] = set()
    sells_map: dict[str, set[str]] = {}
    buys_map:  dict[str, set[str]] = {}
    bu_names_df = None

    if df_bu_companies is not None and not df_bu_companies.empty:
        if "bu_key" not in df_bu_companies.columns:
            raise ValueError("df_bu_companies must include 'bu_key'")
        bu_key_set = set(df_bu_companies["bu_key"].astype(str))
        # allow picking own BU with your shortlist helper (needs a 'name' col)
        bu_names_df = pd.DataFrame({"name": sorted(bu_key_set)})

        # build adjacency with reciprocity checks later
        for _, r in df_bu_companies.iterrows():
            k = str(r["bu_key"])
            sells_map[k] = _parse_refs(r.get("sells_to"))
            buys_map[k]  = _parse_refs(r.get("buys_from"))

    # --- helpers to build candidate pools ---
    def _combined_partner_pool():
        if name_column == "product_name":
            base = df_customers
        else:
            base = df_vendors

        if base is not None and not base.empty:
            if "name" in base.columns:
                base_use = base[["name"]].copy()
            elif "customer_name" in base.columns:
                base_use = base.rename(columns={"customer_name":"name"})[["name"]].copy()
            elif "vendor_name" in base.columns:
                base_use = base.rename(columns={"vendor_name":"name"})[["name"]].copy()
            else:
                base_use = pd.DataFrame(columns=["name"])
        else:
            base_use = pd.DataFrame(columns=["name"])

        if bu_names_df is not None and not bu_names_df.empty:
            return pd.concat([base_use, bu_names_df], ignore_index=True).drop_duplicates("name")
        return base_use

    def _allowed_bu_partners(own_bu: str) -> list[str]:
        """Return BU partner keys allowed by graph + reciprocity."""
        if not own_bu or own_bu not in bu_key_set:
            return []

        if name_column == "product_name":
            # seller → choose from seller.sells_to; require reciprocal buyer.buys_from contains seller
            candidates = sells_map.get(own_bu, set())
            allowed = [p for p in candidates if own_bu in buys_map.get(p, set())]
        else:
            # buyer → choose from buyer.buys_from; require reciprocal seller.sells_to contains buyer
            candidates = buys_map.get(own_bu, set())
            allowed = [p for p in candidates if own_bu in sells_map.get(p, set())]

        # keep only known BUs
        return [p for p in allowed if p in bu_key_set]

    def worker(item_name: str):
        # --- 1) Pick account (same as before) ---
        acc_cands = _shortlist_accounts(df_accounts, ACC_ALLOWED, item_name, k=80) or \
                    _shortlist_accounts(df_accounts, ACC_ALLOWED, "", k=80)

        acc_payload = {
            "task": "Select one GL account by index (normal account, not intercompany).",
            "item_category": name_column,
            "item_name": item_name,
            "candidates": acc_cands,
            "output_schema": {"account_idx": 0},
        }
        acc_idx = _pick_idx_with_gpt(
            client,
            'Return STRICT JSON: {"account_idx": <int>}.',
            acc_payload, len(acc_cands),
            f"{item_name}|{name_column}|account",
            model=model
        )
        acct = acc_cands[acc_idx]
        account_name = acct["name"]

        # --- 2) Pick OWN BU (unchanged) ---
        own_bu = None
        if bu_names_df is not None and not bu_names_df.empty:
            bu_cands = _shortlist_partners(bu_names_df, item_name, account_name, k=50)
            if bu_cands:
                bu_payload = {
                    "task": ("Select the SELLER BU by index" if name_column == "product_name"
                            else "Select the BUYER BU by index"),
                    "item_name": item_name,
                    "account_name": account_name,
                    "candidates": bu_cands,
                    "output_schema": {"bu_idx": 0},
                }
                bu_idx = _pick_idx_with_gpt(
                    client,
                    'Return STRICT JSON: {"bu_idx": <int>}.',
                    bu_payload, len(bu_cands),
                    f"{item_name}|{acct['name']}|own_bu",
                    model=model
                )
                own_bu = bu_cands[bu_idx]["name"]

        # --- 3) Build partner pool and pick ONE external partner ---
        if own_bu:
            allowed_partner_keys = _allowed_bu_partners(own_bu)
        else:
            allowed_partner_keys = []

        if allowed_partner_keys:
            partner_pool = pd.DataFrame({"name": allowed_partner_keys})
        else:
            partner_pool = _combined_partner_pool()

        partner_cands = _shortlist_partners(partner_pool, item_name, account_name, k=200)
        external_cands = [c for c in partner_cands if not str(c["name"]).startswith("BIOCIRC")]

        if external_cands:
            par_payload = {
                "task": "Select one EXTERNAL partner by index (avoid BIOCIRC intercompany).",
                "item_name": item_name,
                "account_name": account_name,
                "candidates": external_cands,
                "output_schema": {"partner_idx": 0},
            }
            pidx = _pick_idx_with_gpt(
                client,
                'Return STRICT JSON: {"partner_idx": <int>}.',
                par_payload, len(external_cands),
                f"{item_name}|{acct['name']}|partner_external",
                model=model
            )
            partner_name = external_cands[pidx]["name"]
        else:
            partner_name = "External Partner"

        # --- Return only ONE row (no intercompany rows here) ---
        return [{
            "name": item_name,
            "account_id": acct["AccountKey"],
            "account_name": account_name,
            "customer_name": partner_name if name_column == "product_name" else None,
            "vendor_name": partner_name if name_column != "product_name" else None,
            "bu_id": own_bu,
        }]

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, str(r["name"]).strip()) for _, r in df_items.iterrows()]
        for f in as_completed(futs):
            rows.extend(f.result())

    return pd.DataFrame(
        rows,
        columns=["name","account_id","account_name","customer_name","bu_id", "vendor_name"]
    )


def map_procurement_services(
    df_procurement: pd.DataFrame,
    df_services: pd.DataFrame,
    df_accounts: pd.DataFrame,
    df_customers: pd.DataFrame,
    df_vendors: pd.DataFrame,
    df_bu_companies: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Combines procurement and service data into a unified spend dataset.

    Returns:
        - df_spend: Combined spend data with item_name, source_type, and *_id columns
        - df_mapping: Combined mapping with item_name and GL/cost center info
    """

    # Create mappings
    procurement_mapping = create_mapping_parallel_simple(
        df_items=df_procurement, df_accounts=df_accounts, df_customers=df_customers, df_vendors=df_vendors, df_bu_companies=df_bu_companies, name_column="procurement_name"
    )
    print("✔ Procurement mapping done!")
    services_mapping = create_mapping_parallel_simple(
        df_items=df_services, df_accounts=df_accounts, df_customers=df_customers, df_vendors=df_vendors, df_bu_companies=df_bu_companies, name_column="service_name"
    )
    print("✔ Service mapping done!")

    # Assign ID and metadata columns
    df_procurement = df_procurement.copy()
    df_procurement["procurement_id"] = df_procurement["name"]
    df_procurement["service_id"] = None
    df_procurement["product_id"] = None
    df_procurement["item_name"] = df_procurement["name"]
    df_procurement["unit_price"] = df_procurement["unit_price"].fillna(0) 
    df_procurement["source_type"] = "procurement"

    df_services = df_services.copy()
    df_services["service_id"] = df_services["name"]
    df_services["procurement_id"] = None
    df_services["product_id"] = None
    df_services["item_name"] = df_services["name"]
    df_services["unit_price"] = df_services["unit_price"].fillna(0)  
    df_services["source_type"] = "service"

    # Combine and select columns
    df_spend = pd.concat([df_procurement, df_services], ignore_index=True)
    df_spend = df_spend[[
        "item_name", "source_type", "annual_spend", "unit_price", "proportionality",
        "product_id", "procurement_id", "service_id"
    ]]

    # Add item_name to mappings for merge compatibility
    procurement_mapping["item_name"] = procurement_mapping["name"]
    services_mapping["item_name"] = services_mapping["name"]

    df_mapping = pd.concat([procurement_mapping, services_mapping], ignore_index=True)
    df_mapping = df_mapping[["item_name", "account_id", "account_name", "vendor_name", "bu_id"]] 
    
    return df_spend, df_mapping

def map_products(
    df_products: pd.DataFrame,
    df_accounts: pd.DataFrame,
    df_customers: pd.DataFrame,
    df_vendors: pd.DataFrame,
    df_bu_companies: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Assign ID and metadata columns
    df_products["product_id"] = df_products["name"]
    df_products["procurement_id"] = None
    df_products["service_id"] = None
    df_products["item_name"] = df_products["name"]
    df_products["source_type"] = "Product Sales"

    # Build spend data
    df_spend = df_products[[
        "item_name", "source_type", "annual_spend", "unit_price", "proportionality",
        "product_id", "procurement_id", "service_id"
    ]]

    # Build mapping
    df_mapping = create_mapping_parallel_simple(
        df_items=df_products,
        df_accounts=df_accounts,
        df_customers=df_customers,
        df_vendors=df_vendors,
        df_bu_companies=df_bu_companies,
        name_column="product_name"
    )

    df_mapping["item_name"] = df_mapping["name"]
    df_mapping = df_mapping[["item_name", "account_id", "account_name", "bu_id", "customer_name"]]
    return df_spend, df_mapping

    
def remap_vendors_customers_with_bu(
    df_customers: pd.DataFrame,
    df_vendors: pd.DataFrame,
    df_bu_companies: pd.DataFrame,
    *,
    ic_range: tuple[float, float] = (0.05, 0.10),   # target Intercompany share
    rng_seed: int | None = 123                      # set None for nondeterministic
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add BU entries as intercompany partners and make them account for a target share
    (5–10% by default) of the 'proportionality' column (which sums to 1).
    """
    rng = np.random.default_rng(rng_seed)
    ic_share = float(rng.uniform(*ic_range))

    # --- Work on copies
    cust = df_customers.copy()
    vend = df_vendors.copy()
    bu   = df_bu_companies.copy()

    # --- Ensure proportionality exists and sums to 1 before scaling
    def _ensure_prop(df: pd.DataFrame, col: str = "proportionality") -> pd.DataFrame:
        if col not in df.columns:
            df[col] = 1.0 / max(len(df), 1)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        s = df[col].sum()
        if s <= 0:
            df[col] = 1.0 / max(len(df), 1)
        else:
            df[col] = df[col] / s
        return df

    cust = _ensure_prop(cust, "proportionality")
    vend = _ensure_prop(vend, "proportionality")

    # --- Build BU partner rows (equal split of the IC share)
    if "bu_key" not in bu.columns or len(bu) == 0:
        raise ValueError("df_bu_companies must contain non-empty 'bu_key' column.")

    bu_rows = pd.DataFrame({
        "name": bu["bu_key"].astype(str)
    })
    bu_n = len(bu_rows)
    bu_share_each = ic_share / bu_n

    bu_rows["proportionality"] = bu_share_each
    bu_rows["customer_segment"] = "Intercompany"
    bu_rows["vendor_segment"]   = "Intercompany"

    # --- Assign new IDs
    start_id_cust = (int(cust["customer_id"].max()) + 1) if "customer_id" in cust.columns and len(cust) else 1
    start_id_vend = (int(vend["vendor_id"].max()) + 1) if "vendor_id" in vend.columns and len(vend) else 1
    bu_rows["customer_id"] = range(start_id_cust, start_id_cust + bu_n)
    bu_rows["vendor_id"]   = range(start_id_vend, start_id_vend + bu_n)

    # propagate IDs back to df_bu_companies (handy for graph lookups later)
    bu["customer_id"] = bu_rows["customer_id"].values
    bu["vendor_id"]   = bu_rows["vendor_id"].values

    # --- Scale existing partners down to (1 - ic_share)
    cust["proportionality"] *= (1.0 - ic_share)
    vend["proportionality"] *= (1.0 - ic_share)

    # --- Merge & final normalize (exactly sum to 1.0)
    cust_out = pd.concat(
        [cust, bu_rows[["name","proportionality","customer_segment","customer_id"]]],
        ignore_index=True
    )
    vend_out = pd.concat(
        [vend, bu_rows[["name","proportionality","vendor_segment","vendor_id"]]],
        ignore_index=True
    )

    # exact re-normalization to guard against float drift
    cust_out["proportionality"] /= cust_out["proportionality"].sum()
    vend_out["proportionality"] /= vend_out["proportionality"].sum()

    return cust_out, vend_out, bu


def pick_intercomp(df_map_products: pd.DataFrame,
                   df_map_expenses: pd.DataFrame,
                   df_bu_companies: pd.DataFrame):
    # 1) Filter to rows where bu_id is in buys_from (valid intercompany candidates)
    valid_idx_prod = df_map_products.index[
        df_map_products["bu_id"].isin(df_bu_companies["buys_from"].values)
    ]
    valid_idx_exp = df_map_expenses.index[
        df_map_expenses["bu_id"].isin(df_bu_companies["buys_from"].values)
    ]

    # Make sure we have enough candidates
    n_pick = min(15, len(valid_idx_prod), len(valid_idx_exp))
    if n_pick == 0:
        raise ValueError("No valid intercompany candidates found!")

    # Pick the SAME random subset from both
    ic_indices = np.random.choice(valid_idx_prod, size=n_pick, replace=False)

    # Precompute pools for performance
    sells_to_pool = df_bu_companies["sells_to"].dropna().tolist()
    buys_from_pool = df_bu_companies["buys_from"].dropna().tolist()

    # 2) Apply to products and expenses
    for idx in ic_indices:
        # --- Products: intercompany sales ---
        df_map_products.loc[idx, "account_id"] = 4007
        df_map_products.loc[idx, "account_name"] = "Inter Company Gross Sales"
        if sells_to_pool:
            df_map_products.loc[idx, "customer_name"] = np.random.choice(sells_to_pool)

        # --- Expenses: intercompany COS ---
        df_map_expenses.loc[idx, "account_id"] = 4009
        df_map_expenses.loc[idx, "account_name"] = "Inter Company COS"
        if buys_from_pool:
            df_map_expenses.loc[idx, "vendor_name"] = np.random.choice(buys_from_pool)
    return df_map_products, df_map_expenses