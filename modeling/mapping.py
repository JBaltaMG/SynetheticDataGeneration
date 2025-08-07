
import random
import numpy as np
import pandas as pd
import os
from utils.utils import create_mapping_from_metadata

def map_procurement_services(
    df_procurement: pd.DataFrame,
    df_services: pd.DataFrame,
    df_accounts: pd.DataFrame,
    df_departments: pd.DataFrame,
    df_customers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines procurement and service data into a unified spend dataset.

    Returns:
        - df_spend: Combined spend data with item_name, source_type, and *_id columns
        - df_mapping: Combined mapping with item_name and GL/cost center info
    """

    # Create mappings

    procurement_mapping = create_mapping_from_metadata(
        df_procurement, df_accounts, df_departments, df_customers, name_column="procurement_name"
    )
    services_mapping = create_mapping_from_metadata(
        df_services, df_accounts, df_departments, df_customers, name_column="service_name"
    )

    # Assign ID and metadata columns
    df_procurement = df_procurement.copy()
    df_procurement["procurement_id"] = df_procurement["name"]
    df_procurement["service_id"] = None
    df_procurement["product_id"] = None
    df_procurement["item_name"] = df_procurement["name"]
    df_procurement["source_type"] = "Procurement"

    df_services = df_services.copy()
    df_services["service_id"] = df_services["name"]
    df_services["procurement_id"] = None
    df_services["product_id"] = None
    df_services["item_name"] = df_services["name"]
    df_services["source_type"] = "Service"

    # Combine and select columns
    df_spend = pd.concat([df_procurement, df_services], ignore_index=True)
    df_spend = df_spend[[
        "item_name", "source_type", "annual_spend", "proportionality",
        "product_id", "procurement_id", "service_id"
    ]]

    # Add item_name to mappings for merge compatibility
    procurement_mapping["item_name"] = procurement_mapping["name"]
    services_mapping["item_name"] = services_mapping["name"]

    df_mapping = pd.concat([procurement_mapping, services_mapping], ignore_index=True)
    df_mapping = df_mapping[["item_name", "account_id", "account_name"]]

    return df_spend, df_mapping

def map_products(
    df_products: pd.DataFrame, 
    df_accounts: pd.DataFrame, 
    df_departments: pd.DataFrame,
    df_customers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Maps product data to GL revenue accounts, departments, and customers.

    Returns:
        - df_spend: Product sales spend with item_name, source_type, and product_id
        - df_mapping: Mapping from product name to GL/cost center/customer info
    """

    df_products = df_products.copy()

    # Assign ID and metadata columns
    df_products["product_id"] = df_products["name"]
    df_products["procurement_id"] = None
    df_products["service_id"] = None
    df_products["item_name"] = df_products["name"]
    df_products["source_type"] = "Product Sales"

    # Build spend data
    df_spend = df_products[[
        "item_name", "source_type", "annual_spend", "proportionality",
        "product_id", "procurement_id", "service_id"
    ]]

    # Build mapping
    df_mapping = create_mapping_from_metadata(
        df=df_products,
        df_accounts=df_accounts,
        df_departments=df_departments,
        df_customers=df_customers,
        name_column="product_name"
    )

    df_mapping["item_name"] = df_mapping["name"]
    df_mapping = df_mapping[["item_name", "account_id", "account_name", "customer_name"]]

    return df_spend, df_mapping