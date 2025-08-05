schemadict = {
    "dim_department": {
        "department_id": "int",
        "department_name": "str",
        "proportionality": "float",
    },
    "dim_customer": {
        "customer_id": "int",
        "CustomerName": "str",
        "CustomerSegment": "str",
    },
    "dim_product": {
        "product_id": "int",
        "product_name": "str",
        "product_category": "str",
        "product_subcategory": "str",
    },
    "dim_account": {
        "Account_ID": "int",
        "AccountName": "str",
        "AccountType": "str",
    },
    "dim_procurement": {
        "ProcurementName": "str",
    },
    "dim_service": {
        "ServiceName": "str",
    },

    "dim_payline": {
        "name": "str",
        "line_id": "int",
    },

    "dim_employee": {
        "role_name": "str",
        "monthly_pay": "float",
        "first_name": "str",
        "last_name": "str",
        "employee_id": "int",
        "department": "str",
    },

    "fact_general_ledger": {
        "document_number": "int",
        "debit_credit": "str",  # 'D' for debit, 'C' for credit
        "date": "str",             
        "amount": "float",
        "account_id": "int",
        "product_id": "int",
        "procurement_id": "int",
        "service_id": "int",
    },

    "fact_payroll": {
        "date": "str",
        "employee_id": "int",
        "line_id": "int",
        "amount": "float",
    }
}