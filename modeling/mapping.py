
import random
import numpy as np
import pandas as pd
import os
from utils.utils_llm import create_mapping_from_metadata
from typing import Dict, Any, List, Optional
import time
import random

import openai

import dotenv

import utils.prompt_utils as prompt_utils
import generators.random_generators as random_generators


from concurrent.futures import ThreadPoolExecutor, as_completed
import re, hashlib, json, dotenv
from openai import OpenAI

def _overlap_tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))

def _shortlist_accounts(df_accounts, allowed, item_name, k=80):
    toks = _overlap_tokens(item_name)
    sub = df_accounts[df_accounts["account_type"] == allowed][["account_id","name","account_type"]].dropna().copy()
    sub["account_id"] = sub["account_id"].astype(str)
    sub["__score"] = sub["name"].astype(str).apply(lambda s: len(toks & _overlap_tokens(s)))
    sub = sub.sort_values("__score", ascending=False).head(k).drop(columns="__score")

    records = []
    for i, r in sub.reset_index(drop=True).iterrows():
        records.append({
            "idx": i,
            "account_id": r["account_id"],
            "name": str(r["name"]),             # ← use the column, not r.name
            "account_type": r["account_type"],
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
                temperature=0.8,
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

def create_mapping_parallel_simple(
    df_items: pd.DataFrame,
    df_accounts: pd.DataFrame,
    df_customers: Optional[pd.DataFrame]=None,
    df_vendors: Optional[pd.DataFrame]=None,
    *,
    name_column: str="product_name",
    model: str="gpt-4.1",
    max_workers: int=12
) -> pd.DataFrame:
    # validate columns
    for req in [["name"]]:
        miss = [c for c in req if c not in df_items.columns]
        if miss: raise ValueError(f"df_items missing {miss}")
    miss = [c for c in ["account_id","name","account_type"] if c not in df_accounts.columns]
    if miss: raise ValueError(f"df_accounts missing {miss}")

    allowed = {"product_name":"Revenue","service_name":"Service Expense","procurement_name":"Product Expense"}[name_column]

    dotenv.load_dotenv()
    api_key = dotenv.get_key(".env","api_key")

    def worker(item_name: str):
        client = OpenAI(api_key=api_key)
        # shortlist
        acc_cands = _shortlist_accounts(df_accounts, allowed, item_name, k=80)
        if not acc_cands:
            # hard fallback to “most common” type if filtering too strict
            acc_cands = _shortlist_accounts(df_accounts, allowed, item_name="", k=80)

        acc_payload = {
            "task":"Select one GL account by index",
            "item_category": name_column,
            "item_name": item_name,
            "candidates": acc_cands,
            "output_schema": {"account_idx": 0},
        }
        acc_idx = _pick_idx_with_gpt(
            client,
            "Return STRICT JSON: {\"account_idx\": <int>}.",
            acc_payload, len(acc_cands),
            f"{item_name}|{name_column}|account",
            model=model
        )
        acct = acc_cands[acc_idx]

        partner_cands = []
        if name_column == "product_name":
            partner_cands = _shortlist_partners(df_customers, item_name, acct["name"], k=200)
            pv = "customer"
        else:
            partner_cands = _shortlist_partners(df_vendors, item_name, acct["name"], k=200)
            pv = "vendor"

        partner_name = None
        if partner_cands:
            par_payload = {
                "task":"Select one partner by index",
                "item_name": item_name,
                "account_name": acct["name"],
                "candidates": partner_cands,
                "output_schema": {"partner_idx": 0},
            }
            pidx = _pick_idx_with_gpt(
                client,
                "Return STRICT JSON: {\"partner_idx\": <int>}.",
                par_payload, len(partner_cands),
                f"{item_name}|{acct['name']}|partner",
                model=model
            )
            partner_name = partner_cands[pidx]["name"]

        return {
            "name": item_name,
            "account_id": acct["account_id"],
            "account_name": acct["name"],
            "customer_name": partner_name if pv=="customer" else None,
            "vendor_name": partner_name if pv=="vendor" else None,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, str(r["name"]).strip()) for _, r in df_items.iterrows()]
        for f in as_completed(futs):
            rows.append(f.result())

    return pd.DataFrame(rows, columns=["name","account_id","account_name","customer_name","vendor_name"])



def map_procurement_services(
    df_procurement: pd.DataFrame,
    df_services: pd.DataFrame,
    df_accounts: pd.DataFrame,
    df_departments: pd.DataFrame,
    df_vendors: pd.DataFrame,
    df_customers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines procurement and service data into a unified spend dataset.

    Returns:
        - df_spend: Combined spend data with item_name, source_type, and *_id columns
        - df_mapping: Combined mapping with item_name and GL/cost center info
    """

    # Create mappings

    procurement_mapping = create_mapping_parallel_simple(
        df_items=df_procurement, df_accounts=df_accounts, df_customers=df_customers, df_vendors=df_vendors, name_column="procurement_name"
    )
    print("✔ Procurement mapping done!")
    services_mapping = create_mapping_parallel_simple(
        df_items=df_services, df_accounts=df_accounts, df_customers=df_customers, df_vendors=df_vendors, name_column="service_name"
    )
    print("✔ Service mapping done!")

    # Assign ID and metadata columns
    df_procurement = df_procurement.copy()
    df_procurement["procurement_id"] = df_procurement["name"]
    df_procurement["service_id"] = None
    df_procurement["product_id"] = None
    df_procurement["item_name"] = df_procurement["name"]
    df_procurement["source_type"] = "procurement"

    df_services = df_services.copy()
    df_services["service_id"] = df_services["name"]
    df_services["procurement_id"] = None
    df_services["product_id"] = None
    df_services["item_name"] = df_services["name"]
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
    df_mapping = df_mapping[["item_name", "account_id", "account_name", "vendor_name"]]

    return df_spend, df_mapping

def map_products(
    df_products: pd.DataFrame, 
    df_accounts: pd.DataFrame, 
    df_departments: pd.DataFrame,
    df_customers: pd.DataFrame,
    df_vendors: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Maps product data to GL revenue accounts, departments, and customers.

    Returns:
        - df_spend: Product sales spend with item_name, source_type, and product_id
        - df_mapping: Mapping from product name to GL/cost center/customer info
    """

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
        name_column="product_name"
    )
    print("✔ Product mapping done!")

    df_mapping["item_name"] = df_mapping["name"]
    df_mapping["unit_price"] = df_products["unit_price"]
    df_mapping = df_mapping[["item_name", "account_id", "account_name", "customer_name"]]

    return df_spend, df_mapping