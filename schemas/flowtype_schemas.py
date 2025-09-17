standard_flow_map = {
    "Revenue": {
        "main": {"debit": ["Accounts Receivable"], "credit": ["Revenue"]},
        "variants": [
            {"debit": ["Accounts Receivable"], "credit": ["Revenue", "VAT Payable"]},
            {"debit": ["Accounts Receivable", "Deferred Revenue"], "credit": ["Revenue"]}
        ]
    },
    "Product Expense": {
        "main": {"debit": ["Product Expense"], "credit": ["Accounts Payable"]},
        "variants": [
            {"debit": ["Product Expense"], "credit": ["Accounts Payable", "VAT Payable"]},
            {"debit": ["Product Expense", "Freight-In"], "credit": ["Accounts Payable"]}
        ]
    },
    "Service Expense": {
        "main": {"debit": ["Service Expense"], "credit": ["Accounts Payable"]},
        "variants": [
            {"debit": ["Service Expense"], "credit": ["Accounts Payable", "VAT Payable"]},
            {"debit": ["Service Expense"], "credit": ["Accounts Payable", "Accrued Liabilities"]}
        ]
    },
    "CashFlow": {
        "main": {"debit": ["Cash"], "credit": ["Accounts Receivable"]},
        "variants": [
            {"debit": ["Cash"], "credit": ["Deferred Revenue"]},
            {"debit": ["Cash"], "credit": ["VAT Payable"]}
        ]
    }
}



