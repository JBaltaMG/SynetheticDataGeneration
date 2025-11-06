

import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))
import pandas as pd
import numpy as np

def create_erp_data_with_growth(
    df: pd.DataFrame,
    year = 2025,
    noise_pct: float = 0.05,
    max_lines_per_doc: int = 25,
    target_qty_per_line: float = 20.0,
    qty_sigma: float = 0.6,
    year_growth_rates: dict = None,  # e.g., {2023: 1.0, 2024: 1.05, 2025: 1.10}
    monthly_weights: list = None,     # e.g., [0.7, 0.7, 0.8, ...] 12 values for each month
    rng = None
) -> pd.DataFrame:
    """
    Create ERP data with year-over-year growth and intra-year seasonality.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns: bu_id, party_id, account_id, unit_price, 
        markdown, category, annual_spend, item_id
    year : int or tuple
        Single year or (year_start, year_end)
    noise_pct : float
        Random noise variation
    max_lines_per_doc : int
        Maximum lines per document
    target_qty_per_line : float
        Target quantity per line
    qty_sigma : float
        Quantity variability
    year_growth_rates : dict, optional
        Growth multipliers per year. Example:
        {2023: 1.0, 2024: 1.05, 2025: 1.10}  # 5% growth 2024, 10% total by 2025
        If None, all years get equal amounts (no growth)
    monthly_weights : list, optional
        12 values (one per month) to control intra-year distribution.
        Example: [0.7, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
        (weak Jan-Feb, strong Nov-Dec)
        Values are relative - they'll be normalized to sum correctly.
        If None, uniform distribution across months.
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    pd.DataFrame
        ERP transaction data with growth and seasonality
        
    Examples
    --------
    # Year-over-year growth: 2023 baseline, 2024 +5%, 2025 +10%
    erp = create_erp_data_with_growth(
        df_revenue, 
        year=(2023, 2025),
        year_growth_rates={2023: 1.0, 2024: 1.05, 2025: 1.10}
    )
    
    # Seasonality: weak Q1, strong Q4
    erp = create_erp_data_with_growth(
        df_revenue,
        year=2025,
        monthly_weights=[0.7, 0.7, 0.8, 0.9, 0.95, 1.0, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4]
    )
    
    # Both growth and seasonality
    erp = create_erp_data_with_growth(
        df_revenue,
        year=(2023, 2025),
        year_growth_rates={2023: 1.0, 2024: 1.08, 2025: 1.15},
        monthly_weights=[0.8, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    )
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Handle year parameter
    if isinstance(year, tuple):
        year_start, year_end = year
        years = list(range(year_start, year_end + 1))
    else:
        years = [year]
    
    # Default growth rates (equal distribution)
    if year_growth_rates is None:
        year_growth_rates = {y: 1.0 for y in years}
    
    # Ensure all years have growth rates
    for y in years:
        if y not in year_growth_rates:
            year_growth_rates[y] = 1.0
    
    # Calculate total growth sum for normalization
    # total_growth_sum = sum(year_growth_rates[y] for y in years)
    
    # Default monthly weights (uniform)
    if monthly_weights is None:
        monthly_weights = [1.0] * 12
    
    # Normalize monthly weights to sum to 12 (so average is 1.0)
    monthly_sum = sum(monthly_weights)
    monthly_weights = [w * 12 / monthly_sum for w in monthly_weights]
    
    # Generate business days for all years
    start_date = pd.Timestamp(f"{years[0]}-01-01")
    end_date = pd.Timestamp(f"{years[-1]}-12-31")
    business_days = pd.bdate_range(start_date, end_date)
    
    rows = []
    doc_counter = 1000
    
    for idx, row in df.iterrows():
        total_annual_amount = row["annual_spend"]
        unit_price = max(row["unit_price"], 1.0)
        category = row["category"]
        
        # Process each year
        for year_val in years:
            # Apply year-specific growth
            year_multiplier = year_growth_rates[year_val]
            year_amount = (total_annual_amount * year_multiplier)
            
            # Get business days for this year
            year_business_days = business_days[business_days.year == year_val]
            
            # Group business days by month
            year_df = pd.DataFrame({'date': year_business_days})
            year_df['month'] = year_df['date'].dt.month
            months_available = year_df.groupby('month')['date'].apply(list).to_dict()
            
            # Distribute amount across months using weights
            monthly_amounts = {}
            for month in range(1, 13):
                if month in months_available:
                    monthly_amounts[month] = (year_amount / 12) * monthly_weights[month - 1]
            
            # Process each month
            for month, month_amount in monthly_amounts.items():
                if month_amount <= 0:
                    continue
                
                # Calculate docs for this month's amount
                target_doc_value = unit_price * min(max_lines_per_doc, 10)
                target_doc_value = max(target_doc_value, 5000)
                
                n_docs = max(1, int(np.ceil(month_amount / target_doc_value)))
                
                # Split month amount across documents
                month_cents = int(month_amount * 100)
                if n_docs == 1:
                    doc_amounts = [month_amount]
                else:
                    doc_cents = rng.multinomial(month_cents, np.ones(n_docs) / n_docs)
                    doc_amounts = doc_cents / 100.0
                
                # Get this month's business days
                month_days = months_available[month]
                
                # Create documents
                for doc_idx, doc_amount in enumerate(doc_amounts):
                    if doc_amount <= 0:
                        continue
                    
                    # Calculate lines
                    z = rng.lognormal(mean=0.0, sigma=qty_sigma)
                    tq = max(1.0, target_qty_per_line * z)
                    
                    expected_lines = doc_amount / (unit_price * tq)
                    expected_lines = max(1.0, min(expected_lines, max_lines_per_doc * 2))
                    
                    if expected_lines < 700:
                        n_lines = max(1, min(max_lines_per_doc, int(rng.poisson(expected_lines))))
                    else:
                        n_lines = max(1, min(max_lines_per_doc, int(rng.normal(expected_lines, np.sqrt(expected_lines)))))
                    
                    # Split document across lines
                    doc_cents = int(doc_amount * 100)
                    if n_lines == 1:
                        line_amounts = [doc_amount]
                    else:
                        line_cents = rng.multinomial(doc_cents, np.ones(n_lines) / n_lines)
                        line_amounts = line_cents / 100.0
                    
                    # Random date from this month
                    doc_date = pd.Timestamp(rng.choice(month_days))
                    
                    # Document number
                    doc_number = f"700-{doc_counter:06d}"
                    doc_counter += 100
                    
                    # Category-specific settings
                    if category == "Revenue":
                        debit_credit = "Credit"
                        sign = 1.0
                        customer_id = row["party_id"]
                        vendor_id = None
                    else:  # COGS
                        debit_credit = "Debit"
                        sign = -1.0
                        customer_id = None
                        vendor_id = row["party_id"]
                    
                    # Create lines
                    for line_amount in line_amounts:
                        if line_amount <= 0:
                            continue
                        
                        qnoise = rng.lognormal(mean=0.0, sigma=noise_pct)
                        quantity = np.round(max(1.0, (line_amount / unit_price) * qnoise)).astype(int)

                        signed_amount = sign * quantity * unit_price

                        transaction = {
                            "document_number": doc_number,
                            "date": doc_date,
                            "bu_id": row["bu_id"],
                            "party_id": row["party_id"],
                            "account_id": row["account_id"],
                            "item_id": row["item_id"],
                            "debit_credit": debit_credit,
                            "amount": signed_amount,
                            "markdown": row["markdown"],
                            "quantity": quantity,
                            "customer_id": customer_id,
                            "vendor_id": vendor_id,
                            "category": category
                        }
                        
                        rows.append(transaction)
    
    # Create DataFrame and sort
    erp_df = pd.DataFrame(rows)
    erp_df = erp_df.sort_values("date").reset_index(drop=True)
    
    return erp_df

def create_budget_from_erp(
    df_erp: pd.DataFrame,
    budget_years: list = None,      # e.g., [2024, 2025] - will use previous year as baseline
    price_growth: float = 0.02,     # 2% price increase
    volume_growth: float = 0.01,    # 1% volume increase  
    noise_pct: float = 0.08,        # 8% random noise
    optimism_bias: float = 0.05,    # Budget typically 5% more optimistic
    rounding: int = 100,            # Round to nearest 100
    budget_day: int = 1,            # Budget date (1st of month)
    doc_prefix: str = "BUDG-",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate budget data from actual ERP data.
    
    Budget uses previous year as baseline and applies:
    - Price growth (e.g., 2% inflation)
    - Volume growth (e.g., 1% more units)
    - Random noise to simulate estimation uncertainty
    - Optional optimism bias (budgets tend to be optimistic)
    
    Parameters
    ----------
    df_erp : pd.DataFrame
        Actual ERP data with columns: date, amount, quantity, account_id, 
        bu_id, party_id, item_id, category, etc.
    budget_years : list, optional
        Years to generate budget for. Uses previous year as baseline.
        E.g., [2024, 2025] uses 2023 for 2024 budget, 2024 for 2025 budget.
        If None, generates for all years after the first.
    price_growth : float
        Expected price increase (e.g., 0.02 = 2%)
    volume_growth : float
        Expected volume increase (e.g., 0.01 = 1%)
    noise_pct : float
        Random noise as percentage (e.g., 0.08 = ±8%)
    optimism_bias : float
        Systematic optimism in budget (e.g., 0.05 = 5% higher)
    rounding : int
        Round amounts to nearest N (e.g., 100)
    budget_day : int
        Day of month for budget dates (typically 1)
    doc_prefix : str
        Prefix for budget document numbers
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Budget data with same structure as ERP data plus 'scenario' column
        
    Examples
    --------
    # Generate budget for 2024 and 2025 using actuals from prior years
    budget = create_budget_from_erp(
        erp_combined,
        budget_years=[2024, 2025],
        price_growth=0.03,      # 3% price increase expected
        volume_growth=0.02,     # 2% volume growth expected
        optimism_bias=0.05      # Budget 5% more optimistic than reality
    )
    """
    rng = np.random.default_rng(seed)
    
    df = df_erp.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    years = sorted(df['year'].unique())
    
    # If no budget years specified, use all years after first
    if budget_years is None:
        budget_years = years[1:]
    
    # Group keys for aggregation
    group_keys = ['year', 'month', 'account_id', 'bu_id', 'party_id', 'item_id', 'category']
    available_keys = [k for k in group_keys if k in df.columns]
    
    budget_frames = []
    
    for budget_year in budget_years:
        baseline_year = budget_year - 1
        
        # Get baseline data from previous year
        baseline = df[df['year'] == baseline_year].copy()
        
        if baseline.empty:
            print(f"Warning: No data for baseline year {baseline_year}, skipping budget {budget_year}")
            continue
        
        # Aggregate to monthly level per dimension
        agg = baseline.groupby(available_keys).agg({
            'amount': 'sum',
            'quantity': 'sum',
            'document_number': 'first',  # Keep one doc number for reference
            'debit_credit': 'first',
            'customer_id': 'first',
            'vendor_id': 'first'
        }).reset_index()
        
        # Calculate unit price from baseline
        agg['unit_price'] = np.where(
            agg['quantity'].abs() > 0,
            agg['amount'] / agg['quantity'],
            0
        )
        
        # Store original sign
        agg['sign'] = np.sign(agg['amount']).replace(0, 1)
        
        # Apply growth and noise
        # Price growth
        new_unit_price = agg['unit_price'] * (1 + price_growth)
        
        # Volume growth with noise
        volume_noise = rng.normal(1 + volume_growth, noise_pct, size=len(agg))
        volume_noise = np.maximum(volume_noise, 0.5)  # Don't go below 50%
        new_quantity = np.round(agg['quantity'] * volume_noise).astype(int)
        
        # Calculate new amount
        # For items with valid unit price, use price × quantity
        # Otherwise, apply combined growth to amount
        has_valid_price = (agg['unit_price'] != 0) & np.isfinite(agg['unit_price'])
        
        new_amount = np.where(
            has_valid_price,
            new_unit_price * new_quantity,
            agg['amount'].abs() * (1 + price_growth) * volume_noise
        )
        
        # Apply optimism bias (budget tends to be optimistic)
        new_amount = new_amount * (1 + optimism_bias)
        
        # Restore original sign (Revenue positive, COGS negative)
        new_amount = new_amount * agg['sign']
        
        # Round amounts
        if rounding and rounding > 1:
            new_amount = (new_amount / rounding).round() * rounding
        
        # Update debit/credit based on new amount
        new_debit_credit = np.where(new_amount >= 0, 'Debit', 'Credit')
        
        # Create budget dates (first day of each month)
        budget_dates = pd.to_datetime({
            'year': budget_year,
            'month': agg['month'],
            'day': budget_day
        })
        
        # Build budget dataframe
        budget_df = agg[available_keys].copy()
        budget_df['date'] = budget_dates
        budget_df['amount'] = np.round(new_amount,0)
        budget_df['quantity'] = np.ceil(np.abs(new_quantity))
        budget_df['debit_credit'] = new_debit_credit
        budget_df['document_number'] = doc_prefix + agg['document_number'].astype(str)
        budget_df['scenario'] = 'Budget'

        # Copy customer/vendor IDs
        if 'customer_id' in agg.columns:
            budget_df['customer_id'] = agg['customer_id']
        if 'vendor_id' in agg.columns:
            budget_df['vendor_id'] = agg['vendor_id']

        budget_frames.append(budget_df)
    
    if not budget_frames:
        return pd.DataFrame()
    
    # Combine all budget years
    budget_all = pd.concat(budget_frames, ignore_index=True)
    
    # Sort by date
    budget_all = budget_all.sort_values(['date', 'document_number']).reset_index(drop=True)
    
    return budget_all