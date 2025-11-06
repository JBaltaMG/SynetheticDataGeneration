

# =============================================================================
# ERP Data Generation Schemas
# Define different business scenarios for synthetic data generation
# =============================================================================

# Schema 1: RECOVERY_GROWTH
# Scenario: Company recovering from downturn, strong growth trajectory
# Use case: Post-crisis recovery, market expansion
# Baseline year: 2024 = 1.0
recovery_growth_schema = {
    "name": "Recovery & Strong Growth",
    "description": "Company recovering from crisis with accelerating growth (2024 baseline)",
    "year_growth_rates": {
        2020: 0.65,    # Deep crisis: -35% from baseline
        2021: 0.75,    # Still struggling: -25% from baseline
        2022: 0.85,    # Recovery starting: -15% from baseline
        2023: 0.95,    # Near baseline: -5% from baseline
        2024: 1.00,    # BASELINE YEAR
        2025: 1.10,    # Growth: +10% from baseline
        2026: 1.20     # Strong growth: +20% from baseline
    },
    "monthly_weights": [
        0.75, 0.8, 0.85,    # Q1: Still recovering, weak start
        0.9, 0.95, 1.0,     # Q2: Gaining momentum
        1.05, 1.1, 1.15,    # Q3: Strong performance
        1.2, 1.25, 1.3      # Q4: Excellent finish
    ],
    "budget_params": {
        "price_growth": 0.04,     # 4% inflation
        "volume_growth": 0.08,    # 8% volume growth
        "optimism_bias": 0.08     # 8% optimistic (growth phase)
    }
}


# Schema 2: STEADY_SEASONAL
# Scenario: Mature stable business with strong seasonality
# Use case: Retail, tourism, seasonal products
# Baseline year: 2024 = 1.0
steady_seasonal_schema = {
    "name": "Steady with Strong Seasonality",
    "description": "Stable growth with pronounced seasonal patterns (2024 baseline)",
    "year_growth_rates": {
        2020: 0.85,    # -15% from baseline
        2021: 0.88,    # -12% from baseline
        2022: 0.92,    # -8% from baseline
        2023: 0.96,    # -4% from baseline
        2024: 1.00,    # BASELINE YEAR
        2025: 1.03,    # +3% from baseline
        2026: 1.06     # +6% from baseline (cumulative)
    },
    "monthly_weights": [
        0.75, 0.8, 0.85,   # Q1: Weak
        0.9, 0.95, 1.0,    # Q2: Below average
        1.15, 1.1, 1.15,   # Q3: Above average  
        1.2, 1.25, 1.3     # Q4: Strong finish (Holiday Season)
    ],
    "budget_params": {
        "price_growth": 0.02,     # 2% inflation
        "volume_growth": 0.02,    # 2% volume growth
        "optimism_bias": 0.03     # 3% optimistic (conservative)
    }
}

# Schema 3: DECLINING_BUSINESS
# Scenario: Business in decline, cost-cutting measures
# Use case: Sunset products, market contraction, restructuring
# Baseline year: 2024 = 1.0
declining_business_schema = {
    "name": "Declining Business",
    "description": "Business experiencing contraction and restructuring (2024 baseline)",
    "year_growth_rates": {
        2020: 1.25,    # Peak years: +25% from baseline
        2021: 1.20,    # Still strong: +20% from baseline
        2022: 1.12,    # Starting to decline: +12% from baseline
        2023: 1.06,    # Continued decline: +6% from baseline
        2024: 1.00,    # BASELINE YEAR
        2025: 0.92,    # Decline: -8% from baseline
        2026: 0.85     # Further decline: -15% from baseline
    },
    "monthly_weights": [
        1.1, 1.05, 1.0,         # Q1: Still relatively strong
        0.95, 0.9, 0.85,        # Q2: Declining
        0.8, 0.75, 0.75,        # Q3: Weak
        0.7, 0.7, 0.65          # Q4: Very weak
    ],
    "budget_params": {
        "price_growth": 0.01,     # 1% price pressure
        "volume_growth": -0.05,   # -5% volume decline
        "optimism_bias": 0.10     # 10% optimistic (denial phase)
    }
}

# Schema 4: UNIFORM_STABLE
# Scenario: Very stable business, minimal seasonality
# Use case: B2B services, subscriptions, utilities
# Baseline year: 2024 = 1.0
uniform_stable_schema = {
    "name": "Uniform & Stable",
    "description": "Minimal seasonality, steady predictable growth (2024 baseline)",
    "year_growth_rates": {
        2020: 0.85,    # -15% from baseline
        2021: 0.89,    # -11% from baseline
        2022: 0.93,    # -7% from baseline
        2023: 0.97,    # -3% from baseline
        2024: 1.00,    # BASELINE YEAR
        2025: 1.04,    # +4% from baseline
        2026: 1.08     # +8% from baseline (cumulative)
    },
    "monthly_weights": [
        0.98, 1.0, 1.02,        # Q1: Very stable
        0.99, 1.0, 1.01,        # Q2: Very stable
        1.0, 0.98, 1.0,         # Q3: Very stable
        1.01, 1.0, 0.99         # Q4: Very stable
    ],
    "budget_params": {
        "price_growth": 0.03,     # 3% inflation
        "volume_growth": 0.01,    # 1% volume growth
        "optimism_bias": 0.02     # 2% optimistic (very conservative)
    }
}

# Schema 5: VOLATILE_CYCLICAL
# Scenario: Highly volatile business with mid-year shifts
# Use case: Commodities, construction, cyclical industries
# Baseline year: 2024 = 1.0
volatile_cyclical_schema = {
    "name": "Volatile & Cyclical",
    "description": "High volatility with mid-year boom and bust cycles (2024 baseline)",
    "year_growth_rates": {
        2020: 1.15,    # Boom: +15% from baseline
        2021: 0.85,    # Bust: -15% from baseline
        2022: 1.10,    # Recovery: +10% from baseline
        2023: 0.90,    # Downturn: -10% from baseline
        2024: 1.00,    # BASELINE YEAR
        2025: 1.12,    # Upturn: +12% from baseline
        2026: 0.95     # Correction: -5% from baseline
    },
    "monthly_weights": [
        0.7, 0.75, 0.8,         # Q1: Weak start
        1.3, 1.4, 1.35,         # Q2: Strong boom
        0.85, 0.8, 0.75,        # Q3: Summer slump
        1.1, 1.2, 1.25          # Q4: Year-end recovery
    ],
    "budget_params": {
        "price_growth": 0.05,     # 5% price volatility
        "volume_growth": 0.03,    # 3% volume growth
        "optimism_bias": 0.12     # 12% optimistic (volatile = uncertain)
    }
}

# =============================================================================
# Schema Registry - Easy access to all schemas
# =============================================================================

SCHEMA_REGISTRY = {
    "recovery_growth": recovery_growth_schema,
    "steady_seasonal": steady_seasonal_schema,
    "declining_business": declining_business_schema,
    "uniform_stable": uniform_stable_schema,
    "volatile_cyclical": volatile_cyclical_schema
}

def get_schema(schema_name: str):
    """
    Get a schema by name.
    
    Parameters
    ----------
    schema_name : str
        One of: 'recovery_growth', 'steady_seasonal', 'declining_business',
        'uniform_stable', 'volatile_cyclical'
    
    Returns
    -------
    dict
        Schema configuration
        
    Examples
    --------
    >>> schema = get_schema('recovery_growth')
    >>> print(schema['description'])
    """
    if schema_name not in SCHEMA_REGISTRY:
        available = ', '.join(SCHEMA_REGISTRY.keys())
        raise ValueError(f"Unknown schema '{schema_name}'. Available: {available}")
    return SCHEMA_REGISTRY[schema_name]

def list_schemas():
    """Print all available schemas with descriptions."""
    print("Available ERP Data Schemas:")
    print("=" * 70)
    for name, schema in SCHEMA_REGISTRY.items():
        print(f"\n{name.upper()}")
        print(f"  Name: {schema['name']}")
        print(f"  Description: {schema['description']}")
        years = schema['year_growth_rates']
        print(f"  Years: 2020-2026 (2024 is baseline at 1.0)")
        print(f"  Growth Pattern:")
        for year in sorted(years.keys()):
            pct_change = (years[year] - 1.0) * 100
            print(f"    {year}: {years[year]:.2f} ({pct_change:+.0f}% from baseline)")
    print("\n" + "=" * 70)