schemadict = {
    "dim_department": {
        "department_id": "int",
        "department_name": "str",
        "proportionality": "float",
        "num_employee": "int",
    },
    "dim_customer": {
        "customer_id": "int",
        "customer_name": "str",
        "customer_segment": "str",
    },

    "dim_vendor": {
        "vendor_id": "int",
        "vendor_name": "str",
        "vendor_type": "str",
    },

    "dim_product": {
        "product_name": "str",
        "unit_price": "str",
    },

    "dim_account": {
        "level3": "int",
        "level4": "int",
        "account_name": "str",
        "level4_name": "str",
        "account_id": "int",
    },

    "dim_procurement": {
        "procurement_name": "str",
    },
    
    "dim_service": {
        "service_name": "str",
    },

    "dim_payline": {
        "line_name": "str",
        "line_id": "int",
    },

    "dim_account_coa": {
        "account_key": "int",
        "name": "str",
        "account_type": "str",
        "ac_sign": "int",
        "budget_area": "str",
        "gl_level_01": "str",
        "gl_level_02": "str",
        "gl_level_03": "str",
        "gl_level_04": "str",
        "gl_level_05": "str",
        "cf_level_01": "str",
        "cf_level_02": "str",
        "cf_level_03": "str",
        "cf_level_04": "str",
        "cf_level_05": "str",
        "cf_level_06": "str",
        "cf_level_07": "str",
        "cf_level_08": "str",
    },

    "dim_bu": {
        "companyname": "str",
        "companykey": "int",
        "bu_name": "str",
        "bu_name_human": "str",
        "bu_type": "str",
        "sells_to": "str",
        "buys_from": "str",
        "customer_id": "int",
        "vendor_id": "int"
    },
    
    "dim_employee": {
        "role_name": "str",
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
        "quantity": "int",
        "account_id": "int",
        "product_id": "int",
        "procurement_id": "int",
        "service_id": "int",
        "vendor_id": "int",
        "customer_id": "int",
        "version": "string",
    },

    "fact_budget": {
        "document_number": "int",
        "debit_credit": "str",  
        "date": "str",             
        "amount": "float",
        "quantity": "int",
        "account_id": "int",
        "product_id": "int",
        "procurement_id": "int",
        "service_id": "int",
        "vendor_id": "int",
        "customer_id": "int"
    },

    "fact_payroll": {
        "date": "str",
        "employee_id": "int",
        "line_id": "int",
        "amount": "float",
    }
}