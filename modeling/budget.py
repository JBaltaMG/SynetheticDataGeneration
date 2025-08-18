import pandas as pd
import numpy as np
from generators.random_generators import generate_dim_date, generate_document_metadata

import pandas as pd
import numpy as np

def generate_budget_from_gl(df_gl: pd.DataFrame, target_year: int, growth: float = 0.05) -> pd.DataFrame:
    """
    Generate a realistic budget fact table from GL actuals.
    
    Args:
        df_gl: Input GL fact table with columns 
               ['document_number','debit_credit','date','amount','account_id',
                'product_id','procurement_id','service_id','vendor_id','customer_id']
        target_year: Budget year to generate
        growth: global growth factor applied to prior-year totals
    """
    
    df = df_gl.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month

    # --- 1. take last full year as baseline ---
    baseline_year = df['Year'].max() - 1
    base = df[df['Year'] == baseline_year]

    # --- 2. aggregate by dimensions + month ---
    dims = ['account_id','product_id','procurement_id','service_id','vendor_id','customer_id']
    agg = base.groupby(dims + ['Month'], dropna=False)['amount'].sum().reset_index()

    # --- 3. calculate seasonality per combo ---
    annual = agg.groupby(dims)['amount'].sum().reset_index().rename(columns={'amount':'annual'})
    season = agg.merge(annual, on=dims, how='left')
    season['weight'] = np.where(season['annual']!=0, season['amount']/season['annual'], 0)

    # --- 4. apply growth ---
    season['annual_budget'] = season['annual'] * (1 + growth)

    # --- 5. distribute annual budget into months ---
    season['budget_amount'] = season['annual_budget'] * season['weight']

    # --- 6. round amounts (nearest 100) ---
    season['budget_amount'] = (season['budget_amount']/100).round()*100

    # --- 7. rebuild fact table format ---
    out = []
    for m in range(1,13):
        month_df = season[season['Month']==m].copy()
        if month_df.empty:
            continue
        # pick a consistent budget date (e.g., 15th of month)
        month_df['date'] = pd.Timestamp(year=target_year, month=m, day=15)
        month_df['document_number'] = [f"BUDG{target_year}{str(m).zfill(2)}{i:04d}" 
                                       for i in range(len(month_df))]
        month_df['debit_credit'] = np.where(month_df['budget_amount']>=0,'Debit','Credit')
        out.append(month_df)

    df_budget = pd.concat(out, ignore_index=True)

    # --- 8. final formatting ---
    df_budget = df_budget[['document_number','debit_credit','date','budget_amount'] + dims]
    df_budget = df_budget.rename(columns={'budget_amount':'amount'})
    df_budget['scenario'] = 'Budget'

    return df_budget
