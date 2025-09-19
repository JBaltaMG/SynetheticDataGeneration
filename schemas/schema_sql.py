# schemas/schema_sql.py
from sqlalchemy.types import (
    Integer, BigInteger, String, Date, DateTime, DECIMAL, Float
)

# Helper aliases
NVARCHAR = String  # on SQL Server this maps to NVARCHAR

schemadict = {
    # ---------- Dimensions ----------
    "dim_department": {
        "id": Integer(),                      # PK
        "department_id": Integer(),
        "department_name": NVARCHAR(255),
        "proportionality": DECIMAL(18, 8),
        "num_employee": Integer(),
        "version_id": Integer(),
    },

    "dim_customer": {
        "id": Integer(),
        "customer_id": Integer(),
        "customer_name": NVARCHAR(255),
        "customer_segment": NVARCHAR(100),
        "version_id": Integer(),
    },

    "dim_vendor": {
        "id": Integer(),
        "vendor_id": Integer(),
        "vendor_name": NVARCHAR(255),
        "vendor_type": NVARCHAR(100),
        "version_id": Integer(),
    },

    "dim_product": {
        "id": Integer(),
        "product_id": Integer(),
        "product_name": NVARCHAR(255),
        "unit_price": DECIMAL(18, 2),         # was "str" → numeric
        "version_id": Integer(),
    },

    "dim_account": {
        "id": Integer(),
        "account_id": Integer(),
        "level3": Integer(),
        "level4": Integer(),
        "account_name": NVARCHAR(255),
        "level4_name": NVARCHAR(255),
        "version_id": Integer(),
    },

    "dim_procurement": {
        "id": Integer(),
        "procurement_id": Integer(),
        "procurement_name": NVARCHAR(255),
        "proportionality": DECIMAL(18, 8),    # commonly needed here
        "unit_price": DECIMAL(18, 2),         # add if you store it here too
        "version_id": Integer(),
    },

    "dim_service": {
        "id": Integer(),
        "service_id": Integer(),
        "service_name": NVARCHAR(255),
        "version_id": Integer(),
    },

    "dim_payline": {
        "id": Integer(),
        "line_id": Integer(),
        "line_name": NVARCHAR(255),
        "version_id": Integer(),
    },

    "dim_account_coa": {
        "id": Integer(),
        "account_key": Integer(),
        "name": NVARCHAR(255),
        "account_type": NVARCHAR(100),
        "ac_sign": Integer(),
        "budget_area": NVARCHAR(255),
        "gl_level_01": NVARCHAR(255),
        "gl_level_02": NVARCHAR(255),
        "gl_level_03": NVARCHAR(255),
        "gl_level_04": NVARCHAR(255),
        "gl_level_05": NVARCHAR(255),
        "cf_level_01": NVARCHAR(255),
        "cf_level_02": NVARCHAR(255),
        "cf_level_03": NVARCHAR(255),
        "cf_level_04": NVARCHAR(255),
        "cf_level_05": NVARCHAR(255),
        "cf_level_06": NVARCHAR(255),
        "cf_level_07": NVARCHAR(255),
        "cf_level_08": NVARCHAR(255),
        "version_id": Integer(),
    },

    "dim_bu": {
        "id": Integer(),
        "companyname": NVARCHAR(255),
        "companykey": Integer(),
        "bu_name": NVARCHAR(255),
        "bu_name_human": NVARCHAR(255),
        "bu_type": NVARCHAR(100),
        "sells_to": NVARCHAR(255),
        "buys_from": NVARCHAR(255),
        "customer_id": Integer(),
        "vendor_id": Integer(),
        "version_id": Integer(),
    },

    "dim_employee": {
        "id": Integer(),
        "employee_id": Integer(),
        "first_name": NVARCHAR(100),
        "last_name": NVARCHAR(100),
        "role_name": NVARCHAR(255),
        "monthly_pay": DECIMAL(18, 2),
        "department_id": Integer(),
        "version_id": Integer(),
    },

    # ---------- Facts ----------
    "fact_general_ledger": {
        "id": BigInteger(),
        "document_number": BigInteger(),      # int64 safe
        "debit_credit": NVARCHAR(10),
        "date": Date(),                       # was "str"
        "amount": DECIMAL(18, 2),
        "quantity": DECIMAL(18, 4),           # was int
        "account_id": Integer(),
        "product_id": Integer(),
        "procurement_id": Integer(),
        "service_id": Integer(),
        "vendor_id": Integer(),
        "customer_id": Integer(),
        "currency": NVARCHAR(10),
        "version_id": Integer(),              # was "version" string
    },

    "fact_budget": {
        "id": BigInteger(),
        "document_number": BigInteger(),
        "debit_credit": NVARCHAR(10),
        "date": Date(),
        "amount": DECIMAL(18, 2),
        "quantity": DECIMAL(18, 4),
        "account_id": Integer(),
        "product_id": Integer(),
        "procurement_id": Integer(),
        "service_id": Integer(),
        "vendor_id": Integer(),
        "customer_id": Integer(),
        "currency": NVARCHAR(10),
        "version_id": Integer(),
    },

    "fact_payroll": {
        "id": BigInteger(),
        "date": Date(),
        "employee_id": Integer(),
        "line_id": Integer(),
        "amount": DECIMAL(18, 2),
        "version_id": Integer(),
    },

    # (and your dim_version table)
    "dim_version": {
        "id": Integer(),
        "version": NVARCHAR(200),
        "created_at": DateTime(),             # if you have it
    },
}