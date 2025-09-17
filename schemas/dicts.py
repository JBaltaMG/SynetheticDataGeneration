# dicts.py
# Centralized schema & mappings for ALL dimensions/facts.
# Import this once anywhere: `import dicts` and call Schema.standardize(df, "Dim_Account") etc.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import pandas as pd

# -------------------------------
# Core data structure
# -------------------------------
@dataclass
class DimensionSchema:
    name: str
    # raw -> canonical
    rename_map: Dict[str, str] = field(default_factory=dict)
    # canonical -> dtype (optional)
    dtypes: Dict[str, str] = field(default_factory=dict)
    # canonical -> pretty label (for BI / exports)
    pretty_labels: Dict[str, str] = field(default_factory=dict)
    # canonical column order to enforce (optional)
    column_order: List[str] = field(default_factory=list)
    # validations
    required_columns: List[str] = field(default_factory=list)
    primary_key: Optional[str] = None
    foreign_keys: Dict[str, str] = field(default_factory=dict)  # col -> referenced dim e.g. "company_id" -> "Dim_Entity"
    # hook for custom post-processing (optional)
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

    def standardize(self, df: pd.DataFrame, *, coerce_dtypes: bool = True, enforce_order: bool = True) -> pd.DataFrame:
        # 1) rename to canonical
        if self.rename_map:
            df = df.rename(columns=self.rename_map)

        # 2) ensure required columns exist
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"[{self.name}] Missing required columns: {missing}")

        # 3) cast dtypes
        if coerce_dtypes and self.dtypes:
            for col, dt in self.dtypes.items():
                if col in df.columns:
                    try:
                        if dt.lower().startswith("int"):
                            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                        elif dt.lower().startswith("float") or dt.lower().startswith("decimal"):
                            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                        elif dt.lower() in ("date", "datetime"):
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                        else:
                            df[col] = df[col].astype("string")
                    except Exception as e:
                        raise ValueError(f"[{self.name}] Failed casting {col} -> {dt}: {e}")

        # 4) column order
        if enforce_order and self.column_order:
            # keep knowns in order, unknowns appended
            known = [c for c in self.column_order if c in df.columns]
            rest = [c for c in df.columns if c not in known]
            df = df[known + rest]

        # 5) custom postprocess
        if self.postprocess is not None:
            df = self.postprocess(df)

        return df


class SchemaRegistry:
    _schemas: Dict[str, DimensionSchema] = {}

    @classmethod
    def register(cls, schema: DimensionSchema):
        cls._schemas[schema.name] = schema

    @classmethod
    def get(cls, name: str) -> DimensionSchema:
        if name not in cls._schemas:
            raise KeyError(f"Schema '{name}' not found. Registered: {list(cls._schemas)}")
        return cls._schemas[name]

    @classmethod
    def standardize(cls, df: pd.DataFrame, name: str, **kwargs) -> pd.DataFrame:
        return cls.get(name).standardize(df, **kwargs)


# -----------------------------------------
# Define per-dimension schemas (start set)
# Fill/adjust maps as your raw names evolve.
# -----------------------------------------

# Dim_Account
SchemaRegistry.register(DimensionSchema(
    name="Dim_Account",
    rename_map={
        # raw -> canonical
        "account_id": "account_id",
        "account_code": "account_code",
        "accountname": "account_name",
        "AccountDescription": "account_description",
        "AccountType": "account_type",
        "AcSign": "account_sign",
        "GLLevel01": "gl_level_01",
        "GLLevel02": "gl_level_02",
        "GLLevel03": "gl_level_03",
        "CFLevel01": "cf_level_01",
        "CFLevel02": "cf_level_02",
    },
    dtypes={
        "account_id": "int",
        "account_code": "string",
        "account_name": "string",
        "account_type": "string",
    },
    pretty_labels={
        "account_id": "Account ID",
        "account_code": "Account Code",
        "account_name": "Account Name",
        "account_type": "Account Type",
    },
    column_order=["account_id","account_code","account_name","account_type","account_description",
                  "account_sign","gl_level_01","gl_level_02","gl_level_03","cf_level_01","cf_level_02"],
    required_columns=["account_code","account_name"],
    primary_key="account_id",
))

# Dim_Product
SchemaRegistry.register(DimensionSchema(
    name="Dim_Product",
    rename_map={
        "product_id": "product_id",
        "ItemCode": "product_code",
        "ItemName": "product_name",
        "Category": "product_category",
        "SubCategory": "product_subcategory",
    },
    dtypes={"product_id": "int", "product_code": "string", "product_name": "string"},
    pretty_labels={"product_name": "Product"},
    column_order=["product_id","product_code","product_name","product_category","product_subcategory"],
    required_columns=["product_code","product_name"],
    primary_key="product_id",
))

# Dim_Customer
SchemaRegistry.register(DimensionSchema(
    name="Dim_Customer",
    rename_map={
        "customer_id": "customer_id",
        "CustomerCode": "customer_code",
        "CustomerName": "customer_name",
        "Region": "region",
        "Country": "country",
        "Industry": "industry",
        "BU": "bu_key",
    },
    dtypes={"customer_id": "int", "customer_code": "string", "customer_name": "string"},
    column_order=["customer_id","customer_code","customer_name","region","country","industry","bu_key"],
    required_columns=["customer_code","customer_name"],
    primary_key="customer_id",
    foreign_keys={"bu_key": "Dim_Entity"},
))

# Dim_Vendor
SchemaRegistry.register(DimensionSchema(
    name="Dim_Vendor",
    rename_map={
        "vendor_id": "vendor_id",
        "VendorCode": "vendor_code",
        "VendorName": "vendor_name",
        "Country": "country",
        "Category": "vendor_category",
    },
    dtypes={"vendor_id": "int"},
    column_order=["vendor_id","vendor_code","vendor_name","country","vendor_category"],
    required_columns=["vendor_code","vendor_name"],
    primary_key="vendor_id",
))

# Dim_Entity (Company + BU)
SchemaRegistry.register(DimensionSchema(
    name="Dim_Entity",
    rename_map={
        "entity_id": "entity_id",
        "entity_key": "entity_key",
        "entity_name": "entity_name",
        "entity_type": "entity_type",           # Company, BusinessUnit, Dept, CostCenter
        "parent_entity_id": "parent_entity_id",
        "company_id": "company_id",
        "currency": "currency",
        "fiscal_start_month": "fiscal_start_month",
        "region": "region",
    },
    dtypes={"entity_id": "int", "parent_entity_id": "int", "company_id": "int", "fiscal_start_month": "int"},
    column_order=["entity_id","entity_key","entity_name","entity_type","parent_entity_id","company_id","currency","fiscal_start_month","region"],
    required_columns=["entity_key","entity_name","entity_type"],
    primary_key="entity_id",
))

# Dim_Office
SchemaRegistry.register(DimensionSchema(
    name="Dim_Office",
    rename_map={
        "office_id": "office_id",
        "OfficeCode": "office_code",
        "OfficeName": "office_name",
        "City": "city",
        "Country": "country",
        "entity_id": "entity_id",  # FK to Dim_Entity (BU/Company)
    },
    dtypes={"office_id": "int", "entity_id": "int"},
    column_order=["office_id","office_code","office_name","city","country","entity_id"],
    required_columns=["office_code","office_name"],
    primary_key="office_id",
    foreign_keys={"entity_id": "Dim_Entity"},
))

# Dim_Employee
SchemaRegistry.register(DimensionSchema(
    name="Dim_Employee",
    rename_map={
        "employee_id": "employee_id",
        "EmployeeNo": "employee_code",
        "FirstName": "first_name",
        "LastName": "last_name",
        "Role": "role_name",
        "entity_id": "entity_id",
    },
    dtypes={"employee_id": "int", "entity_id": "int"},
    column_order=["employee_id","employee_code","first_name","last_name","role_name","entity_id"],
    required_columns=["employee_code","first_name","last_name"],
    primary_key="employee_id",
    foreign_keys={"entity_id": "Dim_Entity"},
))

# Dim_Date
SchemaRegistry.register(DimensionSchema(
    name="Dim_Date",
    rename_map={
        "date": "date",
        "Year": "year",
        "Month": "month",
        "Quarter": "quarter",
        "IsBusinessDay": "is_business_day",
        "IsMonthEnd": "is_month_end",
        "Weekday": "weekday",
    },
    dtypes={"date": "date", "year": "int", "month": "int", "quarter": "int"},
    column_order=["date","year","quarter","month","weekday","is_business_day","is_month_end"],
    required_columns=["date"],
    primary_key=None,
))

# Dim_Currency
SchemaRegistry.register(DimensionSchema(
    name="Dim_Currency",
    rename_map={
        "currency": "currency",
        "CurrencyName": "currency_name",
        "Symbol": "symbol",
    },
    dtypes={"currency": "string"},
    column_order=["currency","currency_name","symbol"],
    required_columns=["currency"],
    primary_key="currency",
))

# Dim_FX (daily/monthly rates)
SchemaRegistry.register(DimensionSchema(
    name="Dim_FX",
    rename_map={
        "date": "date",
        "from_currency": "from_currency",
        "to_currency": "to_currency",
        "rate": "fx_rate",
    },
    dtypes={"date": "date", "fx_rate": "float"},
    column_order=["date","from_currency","to_currency","fx_rate"],
    required_columns=["date","from_currency","to_currency","fx_rate"],
    primary_key=None,
))

# Bridge_EntityHierarchy
SchemaRegistry.register(DimensionSchema(
    name="Bridge_EntityHierarchy",
    rename_map={
        "parent_entity_id": "parent_entity_id",
        "child_entity_id": "child_entity_id",
        "relationship_type": "relationship_type",
    },
    dtypes={"parent_entity_id": "int", "child_entity_id": "int"},
    column_order=["parent_entity_id","child_entity_id","relationship_type"],
    required_columns=["parent_entity_id","child_entity_id"],
    primary_key=None,
    foreign_keys={"parent_entity_id": "Dim_Entity", "child_entity_id": "Dim_Entity"},
))

# Fact_General_Ledger (bonus: keep here so you can reuse)
SchemaRegistry.register(DimensionSchema(
    name="Fact_General_Ledger",
    rename_map={
        "document_number": "document_number",
        "debit_credit": "debit_credit",
        "date": "date",
        "amount": "amount",
        "quantity": "quantity",
        "currency": "currency",
        "account_id": "account_id",
        "product_id": "product_id",
        "procurement_id": "procurement_id",
        "service_id": "service_id",
        "vendor_id": "vendor_id",
        "customer_id": "customer_id",
        "company_id": "company_id",
        "bu_id": "bu_id",
        "doc_link_id": "doc_link_id",
        "id": "id",
        "version_id": "version_id",
    },
    dtypes={
        "date": "date",
        "amount": "float",
        "quantity": "float",
        "account_id": "string", 
        "company_id": "int",
        "bu_id": "int",
        "version_id": "int",
    },
    column_order=["id","version_id","document_number","date","debit_credit","currency","amount","quantity",
                  "account_id","product_id","procurement_id","service_id","vendor_id","customer_id",
                  "company_id","bu_id","doc_link_id"],
    required_columns=["document_number","date","debit_credit","amount","account_id","company_id","bu_id"],
    primary_key="id",
    foreign_keys={"company_id": "Dim_Entity", "bu_id": "Dim_Entity", "account_id": "Dim_Account"},
))

# -------------------------------
# Convenience helpers
# -------------------------------
def standardize(df: pd.DataFrame, schema_name: str, **kwargs) -> pd.DataFrame:
    """Shortcut: dicts.standardize(df, 'Dim_Account')"""
    return SchemaRegistry.standardize(df, schema_name, **kwargs)

def pretty_labels(schema_name: str) -> Dict[str, str]:
    """Return canonical -> pretty label mapping for BI."""
    return SchemaRegistry.get(schema_name).pretty_labels

def get_schema(schema_name: str) -> DimensionSchema:
    return SchemaRegistry.get(schema_name)
