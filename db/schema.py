schemadict = {
    "dim_department": {
        "department_id": "int",
        "department_name": "str",
        "proportionality": "float",
        "numemployees": "int",
    },
    "dim_customer": {
        "customer_id": "int",
        "customer_name": "str",
        "customer_segment": "str",
    },
    "dim_product": {
        "product_id": "int",
        "product_name": "str",
        "product_category": "str",
        "product_subcategory": "str",
    },
    "dim_account": {
        "account_id": "int",
        "account_name": "str",
        "account_type": "str",
    },
    "dim_procurement": {
        "procurement_name": "str",
    },
    "dim_service": {
        "service_name": "str",
    },

    "dim_line": {
        "line_name": "str",
        "line_id": "int",
    },

    "dim_employee": {
        "name": "str",
        "monthly_pay": "float",
        "first_name": "str",
        "last_name": "str",
        "employee_id": "int",
        "department_id": "int",
    },

    "fact_general_ledger": {
        "document_number": "int",
        "debit_credit": "str",  
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