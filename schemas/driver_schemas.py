schema_driver_families = {
    "Manufacturing / Production": {
        "main_volume_driver": "Tonnage",
        "secondary_driver": "Unit price",
        "revenue_logic": "volume * price",
        "cogs_logic": "volume * cost_per_ton",
    },
    "Retail / Consumer": {
        "main_volume_driver": "Units sold",
        "secondary_driver": "Retail price",
        "revenue_logic": "units * price",
        "cogs_logic": "units * production_cost * (1 - margin_pct)",
    },
    "Service / SaaS": {
        "main_volume_driver": "Hours billed",
        "secondary_driver": "Hourly rate",
        "revenue_logic": "hours * rate",
        "cogs_logic": "hours * cost_rate",
    },
    "Trading / Resale": {
        "main_volume_driver": "Quantity traded",
        "secondary_driver": "Market price",
        "revenue_logic": "quantity * sell_price",
        "cogs_logic": "quantity * buy_price",
    },
    "Energy / Utility": {
        "main_volume_driver": "MWh or tons",
        "secondary_driver": "Market rate",
        "revenue_logic": "volume * spot_price",
        "cogs_logic": "volume * feedstock_cost",
    },
}