import pandas as pd
import numpy as np
from generators.random_generators import generate_dim_date, generate_document_metadata

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def generate_monthly_budget(
    df_gl: pd.DataFrame,
    price_growth=0.02,                 # can be float or dict {year: growth}
    volume_growth=0.01,                # can be float or dict {year: growth}
    rounding: int = 100,               # round amount to nearest N (e.g., 100)
    budget_day: int = 1,              # date day for budget
    doc_prefix: str = "BUDG",          # prefix added to original doc number
    seed: int = 42,
    noise_std_amount: float = 0.08,    # ~8% normal noise on amount
    noise_std_qty: float = 0.05,       # ~5% normal noise on quantity
    season_drift: float = 0.15,        # blend toward global seasonality (0..1)
    shock_prob: float = 0.08,          # chance of a random shock per (dims, month)
    shock_range=(0.85, 1.25),          # shock multiplier range
    credit_negative: bool = True       # if True, keep credits negative (like revenue/COGS)
) -> pd.DataFrame:
    """
    Generate a realistic budget for *all years* using the previous year as baseline.
    Adds price/volume effects, seasonality drift, random noise, and occasional shocks.
    Reuses the *original* document numbers with a prefix.

    Input df_gl columns (strings are fine):
      ['document_number','debit_credit','date','amount','quantity',
       'account_id','product_id','procurement_id','service_id','vendor_id','customer_id']

    Output columns:
      ['document_number','debit_credit','date','amount','quantity',
       'account_id','product_id','procurement_id','service_id','vendor_id','customer_id','scenario']
    """
    rng = np.random.default_rng(seed)
    df = df_gl.copy()

    # Basic prep
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month

    df['unit_price'] = df['amount'] / df['quantity'].replace(0, np.nan)

    dims = ['account_id','product_id','procurement_id','service_id','vendor_id','customer_id', 'bu_id']
    base_keys = ['document_number','Month', *dims]  # doc-level month packet

    years = sorted(df['Year'].unique())
    if len(years) < 2:
        return pd.DataFrame(columns=[
            'document_number','debit_credit','date','amount','quantity', *dims, 'scenario'
        ])

    # helpers for per-year growth
    def _get_growth(g, y):
        if isinstance(g, dict):
            # last provided value as fallback if year not in dict
            return float(g.get(y, list(g.values())[-1] if g else 0.0))
        return float(g)

    # GLOBAL MONTH SEASONALITY (signed)
    global_month = (df.groupby('Month', dropna=False)['amount']
                      .sum().rename('amt').reset_index())
    gtot = global_month['amt'].sum()
    if gtot == 0:
        global_month['g_w'] = 1.0 / 12
    else:
        w = global_month['amt'] / gtot
        # normalize to sum 1 (robust in case of rounding)
        global_month['g_w'] = (w / w.sum()).fillna(1.0/12)
    g_w = dict(zip(global_month['Month'], global_month['g_w']))

    out_all_years = []

    # iterate each target year that has a baseline year present
    for target_year in years:
        baseline_year = target_year
        base = df.loc[df['Year'] == baseline_year].copy()
        if base.empty:
            continue

        # Aggregate to doc-month level (keep doc numbers for reuse)
        agg = (base.groupby(base_keys, dropna=False)
                    .agg(amount=('amount','sum'),
                         quantity=('quantity','sum'))  # sum qty across lines in same doc
                    .reset_index())

        # Annual totals per dims (exclude doc_number to learn seasonality by mix)
        annual = (agg.groupby(dims, dropna=False)[['amount','quantity']]
                     .sum().reset_index()
                     .rename(columns={'amount':'annual_amt','quantity':'annual_qty'}))

        # Monthly totals per dims
        month_d = (agg.groupby(dims + ['Month'], dropna=False)[['amount','quantity']]
                      .sum().reset_index()
                      .rename(columns={'amount':'month_amt','quantity':'month_qty'}))

        # Merge to compute *baseline* seasonality weights per dims
        seas = month_d.merge(annual, on=dims, how='left')
        # amount-based weights (fallback to global)
        seas['w_amt_raw'] = np.where(seas['annual_amt'].replace(0,np.nan).notna(),
                                     seas['month_amt'] / seas['annual_amt'].replace(0,np.nan),
                                     np.nan)
        # normalize by dims
        def _norm_grp(s):
            s = s.astype(float)
            sm = s.sum(skipna=True)
            if not np.isfinite(sm) or sm == 0:
                return pd.Series(np.nan, index=s.index)
            return s / sm
        seas['grp_id'] = pd.factorize(seas[dims].apply(lambda r: tuple(r.values), axis=1))[0]
        seas['w_amt'] = (seas.groupby('grp_id')['w_amt_raw'].transform(_norm_grp)
                         .fillna(seas['Month'].map(g_w)))  # fallback to global

        # DRIFT seasonality toward global + little randomness
        # blend: (1 - season_drift)*local + season_drift*global, then renormalize per dims
        seas['w_drift'] = (1 - season_drift) * seas['w_amt'] + season_drift * seas['Month'].map(g_w)
        # tiny random bumps per (dims, month)
        eps = rng.normal(0, 0.02, size=len(seas))  # 2% bumps
        seas['w_drift'] = np.maximum(seas['w_drift'] * (1 + eps), 0.0)
        # renormalize w_drift per dims
        seas['w_drift'] = (seas.groupby('grp_id')['w_drift']
                                .transform(lambda s: s / max(s.sum(), 1e-9)))

        # Build a lookup multiplier per (dims, month) relative to baseline month weight.
        # If baseline weight was zero but drift weight > 0, we’ll still only apply
        # to docs that happen to be in that month (we don’t move docs between months).
        seas['w_ratio'] = np.where(seas['w_amt'] > 0, seas['w_drift'] / seas['w_amt'], 1.0)
        w_ratio = seas.set_index(dims + ['Month'])['w_ratio'].to_dict()

        # Shock table per (dims, month)
        shock_mask = rng.random(len(seas)) < shock_prob
        shock_mult = np.ones(len(seas))
        shock_vals = rng.uniform(shock_range[0], shock_range[1], size=shock_mask.sum())
        shock_mult[shock_mask] = shock_vals
        seas['shock'] = shock_mult
        shock_lookup = seas.set_index(dims + ['Month'])['shock'].to_dict()

        # Merge back to doc-level agg to compute budget
        agg['sign'] = np.sign(agg['amount']).replace(0, 1)  # preserve sign pattern
        # baseline unit price (if quantity available and nonzero)
        agg['unit_price'] = np.where(agg['quantity'].abs() > 0,
                                     agg['amount'] / agg['quantity'],
                                     np.nan)

        # Per-year growth
        pg = _get_growth(price_growth, target_year)
        vg = _get_growth(volume_growth, target_year)

        # Noise draws per row
        amt_noise = rng.normal(0, noise_std_amount, size=len(agg))
        qty_noise = rng.normal(0, noise_std_qty, size=len(agg))

        # Seasonality ratio and shock per row
        # (if not found, default 1.0)
        def _lookup(dic, row, default=1.0):
            key = (row['account_id'], row['product_id'], row['procurement_id'],
                   row['service_id'], row['vendor_id'], row['customer_id'], row['Month'])
            return dic.get(key, default)

        ratios = []
        shocks = []
        for _, r in agg.iterrows():
            ratios.append(_lookup(w_ratio, r, 1.0))
            shocks.append(_lookup(shock_lookup, r, 1.0))
        ratios = np.array(ratios, dtype=float)
        shocks = np.array(shocks, dtype=float)

        # Quantity path (if baseline quantity present)
        qty_base = agg['quantity'].astype(float).fillna(0.0).values
        qty_factor = (1.0 + vg) * (1.0 + qty_noise)
        qty_budget = qty_base * qty_factor

        # Price path (if unit price known)
        up_base = agg['unit_price'].astype(float).values
        up_budget = np.where(np.isfinite(up_base), up_base * (1.0 + pg), np.nan)

        # Amount via price×volume if both known; else fallback to amount growth
        amt_base = agg['amount'].astype(float).values
        amt_growth_factor = (1.0 + pg) * (1.0 + vg) * (1.0 + amt_noise)

        amt_pv = np.where(np.isfinite(up_budget) & (qty_budget != 0),
                          up_budget * qty_budget,
                          np.nan)
        amt_new = np.where(np.isfinite(amt_pv), amt_pv, amt_base * amt_growth_factor)

        # Apply seasonality drift and shocks (scale magnitudes, keep prior sign)
        amt_new = np.abs(amt_new) * ratios * shocks
        # Reapply original sign style (credit_negative keeps credits negative)
        if credit_negative:
            # keep the sign of the baseline amount
            amt_new = agg['sign'].values * amt_new
        else:
            # if you invert elsewhere, adjust here
            pass

        # Round amount
        if rounding and rounding > 1:
            amt_new = (amt_new / rounding).round() * rounding

        # Debit/Credit from sign
        dc = np.where(amt_new >= 0, 'Debit', 'Credit')

        # Quantity rounding (optional): don't round by default
        qty_out = np.where(qty_base != 0, qty_budget, np.nan)

        # Assemble the year frame
        df_year = agg[[*base_keys]].copy()
        df_year['amount'] = amt_new
        df_year['quantity'] = qty_out
        df_year['debit_credit'] = dc
        df_year['date'] = pd.to_datetime({
            'year': target_year,
            'month': df_year['Month'],
            'day': budget_day
        })

        # Reuse original doc number with prefix
        df_year['document_number'] = doc_prefix + df_year['document_number'].astype(str)

        # Final shape
        df_year = df_year[['document_number','debit_credit','date','amount','quantity', *dims]]
        df_year['scenario'] = 'Budget'

        out_all_years.append(df_year)

    if not out_all_years:
        return pd.DataFrame(columns=[
            'document_number','debit_credit','date','amount','quantity', *dims
        ])

    out = pd.concat(out_all_years, ignore_index=True)

    # Optional sanity: if any exact zeros cause 'Credit' with +0; normalize to 'Debit' for 0
    zero_mask = out['amount'] == 0
    out.loc[zero_mask, 'debit_credit'] = 'Debit'

    out['amount'] = out['amount']
    out['quantity'] = abs(np.round(out['quantity']))

    # Sort for readability
    cols = ['document_number','debit_credit','date','amount','quantity', 'account_id','product_id','procurement_id','service_id','vendor_id','customer_id','bu_id']
    out = out[cols] 

    return out[cols]


def generate_weekly_budget(
    df_gl: pd.DataFrame,
    price_growth=0.02,                 # float or dict {year: growth}
    volume_growth=0.01,                # float or dict {year: growth}
    rounding: int = 100,               # round amount to nearest N (e.g., 100)
    doc_prefix: str = "BUDG",          # prefix added to original doc number
    seed: int = 42,
    noise_std_amount: float = 0.08,    # ~8% normal noise on amount
    noise_std_qty: float = 0.05,       # ~5% normal noise on quantity
    season_drift: float = 0.15,        # blend toward global seasonality (0..1)
    shock_prob: float = 0.08,          # chance of a random shock per (dims, month)
    shock_range=(0.85, 1.25),          # shock multiplier range
    credit_negative: bool = True,      # if True, keep credits negative
    week_anchor: str = "W-MON",        # weekly label (period end on Monday)
) -> pd.DataFrame:
    """
    Generate a realistic budget for *all years*, then LUMP INTO WEEKLY buckets.
    Splits each (document_number, Month, dims) monthly amount across the ISO weeks
    overlapping that month, by day-count within each week.

    Input df_gl columns:
      ['document_number','debit_credit','date','amount','quantity',
       'account_id','product_id','procurement_id','service_id','vendor_id','customer_id','bu_id']

    Output columns:
      ['document_number','debit_credit','date','amount','quantity',
       'account_id','product_id','procurement_id','service_id','vendor_id','customer_id','bu_id']
    """
    rng = np.random.default_rng(seed)
    df = df_gl.copy()

    # ---- Prep
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['unit_price'] = df['amount'] / df['quantity'].replace(0, np.nan)

    dims = ['account_id','product_id','procurement_id','service_id','vendor_id','customer_id','bu_id']
    base_keys = ['document_number','Month', *dims]

    years = sorted(df['Year'].unique())
    if len(years) == 0:
        return pd.DataFrame(columns=[
            'document_number','debit_credit','date','amount','quantity', *dims
        ])

    # helpers for per-year growth
    def _get_growth(g, y):
        if isinstance(g, dict):
            return float(g.get(y, list(g.values())[-1] if g else 0.0))
        return float(g)

    # ---- Global month seasonality (signed)
    global_month = df.groupby('Month', dropna=False)['amount'].sum().rename('amt').reset_index()
    gtot = global_month['amt'].sum()
    if gtot == 0:
        global_month['g_w'] = 1.0 / 12
    else:
        w = global_month['amt'] / gtot
        global_month['g_w'] = (w / w.sum()).fillna(1.0/12)
    g_w = dict(zip(global_month['Month'], global_month['g_w']))

    out_all_years = []

    for target_year in years:
        base = df.loc[df['Year'] == target_year].copy()
        if base.empty:
            continue

        # Aggregate to doc-month level (keep doc numbers for reuse)
        agg = (base.groupby(base_keys, dropna=False)
                    .agg(amount=('amount','sum'),
                         quantity=('quantity','sum'))
                    .reset_index())

        # Annual totals per dims (exclude doc_number)
        annual = (agg.groupby(dims, dropna=False)[['amount','quantity']]
                     .sum().reset_index()
                     .rename(columns={'amount':'annual_amt','quantity':'annual_qty'}))

        # Monthly totals per dims
        month_d = (agg.groupby(dims + ['Month'], dropna=False)[['amount','quantity']]
                      .sum().reset_index()
                      .rename(columns={'amount':'month_amt','quantity':'month_qty'}))

        # Local seasonality weights with fallback to global
        seas = month_d.merge(annual, on=dims, how='left')
        seas['w_amt_raw'] = np.where(seas['annual_amt'].replace(0,np.nan).notna(),
                                     seas['month_amt'] / seas['annual_amt'].replace(0,np.nan),
                                     np.nan)

        def _norm_grp(s):
            s = s.astype(float)
            sm = s.sum(skipna=True)
            if not np.isfinite(sm) or sm == 0:
                return pd.Series(np.nan, index=s.index)
            return s / sm

        seas['grp_id'] = pd.factorize(seas[dims].apply(lambda r: tuple(r.values), axis=1))[0]
        seas['w_amt'] = (seas.groupby('grp_id')['w_amt_raw'].transform(_norm_grp)
                         .fillna(seas['Month'].map(g_w)))

        # Drift toward global + tiny noise, then renormalize
        seas['w_drift'] = (1 - season_drift) * seas['w_amt'] + season_drift * seas['Month'].map(g_w)
        eps = rng.normal(0, 0.02, size=len(seas))
        seas['w_drift'] = np.maximum(seas['w_drift'] * (1 + eps), 0.0)
        seas['w_drift'] = (seas.groupby('grp_id')['w_drift']
                                .transform(lambda s: s / max(s.sum(), 1e-9)))

        seas['w_ratio'] = np.where(seas['w_amt'] > 0, seas['w_drift'] / seas['w_amt'], 1.0)
        w_ratio = seas.set_index(dims + ['Month'])['w_ratio'].to_dict()

        # Shocks
        shock_mask = rng.random(len(seas)) < shock_prob
        shock_mult = np.ones(len(seas))
        shock_vals = rng.uniform(shock_range[0], shock_range[1], size=shock_mask.sum())
        shock_mult[shock_mask] = shock_vals
        seas['shock'] = shock_mult
        shock_lookup = seas.set_index(dims + ['Month'])['shock'].to_dict()

        # Baselines per row
        agg['sign'] = np.sign(agg['amount']).replace(0, 1)
        agg['unit_price'] = np.where(agg['quantity'].abs() > 0,
                                     agg['amount'] / agg['quantity'], np.nan)

        pg = _get_growth(price_growth, target_year)
        vg = _get_growth(volume_growth, target_year)

        amt_noise = rng.normal(0, noise_std_amount, size=len(agg))
        qty_noise = rng.normal(0, noise_std_qty, size=len(agg))

        def _lookup(dic, row, default=1.0):
            key = (row['account_id'], row['product_id'], row['procurement_id'],
                   row['service_id'], row['vendor_id'], row['customer_id'], row['Month'])
            return dic.get(key, default)

        ratios = []
        shocks = []
        for _, r in agg.iterrows():
            ratios.append(_lookup(w_ratio, r, 1.0))
            shocks.append(_lookup(shock_lookup, r, 1.0))
        ratios = np.array(ratios, dtype=float)
        shocks = np.array(shocks, dtype=float)

        qty_base = agg['quantity'].astype(float).fillna(0.0).values
        qty_factor = (1.0 + vg) * (1.0 + qty_noise)
        qty_budget = qty_base * qty_factor

        up_base = agg['unit_price'].astype(float).values
        up_budget = np.where(np.isfinite(up_base), up_base * (1.0 + pg), np.nan)

        amt_base = agg['amount'].astype(float).values
        amt_growth_factor = (1.0 + pg) * (1.0 + vg) * (1.0 + amt_noise)

        amt_pv = np.where(np.isfinite(up_budget) & (qty_budget != 0),
                          up_budget * qty_budget,
                          np.nan)
        amt_new = np.where(np.isfinite(amt_pv), amt_pv, amt_base * amt_growth_factor)

        # Apply seasonality drift + shocks to magnitudes, then reapply sign style
        amt_new = np.abs(amt_new) * ratios * shocks
        if credit_negative:
            amt_new = agg['sign'].values * amt_new

        # Round amount
        if rounding and rounding > 1:
            amt_new = (amt_new / rounding).round() * rounding

        # Quantity output (no rounding by default)
        qty_out = np.where(qty_base != 0, qty_budget, np.nan)

        # ---------- WEEKLY LUMPING (split each month across weeks) ----------
        # Build per-row month start/end within the target year/month
        month_start = pd.to_datetime({
            'year': target_year,
            'month': agg['Month'],
            'day': 1
        })
        month_end = (month_start + pd.offsets.MonthEnd(0))

        rows = []
        for i, r in agg.iterrows():
            amt = float(amt_new[i])
            qty = float(qty_out[i]) if np.isfinite(qty_out[i]) else np.nan
            sign = r['sign']

            ms = month_start.iloc[i]
            me = month_end.iloc[i]

            # Make daily range across the month (calendar days)
            days = pd.date_range(ms, me, freq="D")
            if len(days) == 0:
                continue

            # Weekly labels for each day
            week_labels = days.to_series().dt.to_period(week_anchor).dt.end_time.dt.normalize()

            # Count days per week in this month
            w_counts = week_labels.value_counts().sort_index()
            total_days = w_counts.sum()
            if total_days == 0:
                continue

            # Proportional allocation by day-count
            for week_end_date, dcount in w_counts.items():
                w_share = dcount / total_days
                amt_w = amt * w_share
                qty_w = qty * w_share if np.isfinite(qty) else np.nan

                # Debit/Credit from sign of allocated amount (after allocation it can be +/- 0)
                dc = 'Debit' if amt_w >= 0 else 'Credit'

                # Unique weekly doc number
                iso_year = pd.Timestamp(week_end_date).isocalendar().year
                iso_week = pd.Timestamp(week_end_date).isocalendar().week
                docnum = f"{doc_prefix}{r['document_number']}-W{iso_year}{int(iso_week):02d}"

                rows.append({
                    'document_number': docnum,
                    'debit_credit': dc if amt_w != 0 else 'Debit',
                    'date': pd.Timestamp(week_end_date),
                    'amount': amt_w,
                    'quantity': qty_w,
                    'account_id': r['account_id'],
                    'product_id': r['product_id'],
                    'procurement_id': r['procurement_id'],
                    'service_id': r['service_id'],
                    'vendor_id': r['vendor_id'],
                    'customer_id': r['customer_id'],
                    'bu_id': r['bu_id'],
                })

        df_year = pd.DataFrame(rows)
        out_all_years.append(df_year)

    if not out_all_years:
        return pd.DataFrame(columns=[
            'document_number','debit_credit','date','amount','quantity', *dims
        ])

    out = pd.concat(out_all_years, ignore_index=True)

    # Normalize exact zeros to Debit for neatness
    zero_mask = out['amount'] == 0
    out.loc[zero_mask, 'debit_credit'] = 'Debit'
    out['amount'] = out['amount'].round(2)
    # Sort & final columns
    cols = ['document_number','debit_credit','date','amount','quantity',
            'account_id','product_id','procurement_id','service_id','vendor_id','customer_id','bu_id']
    out = out[cols].sort_values(['bu_id','date','document_number']).reset_index(drop=True)
    return out

def generate_difference_analysis(
    df_actuals: pd.DataFrame,
    df_budget: pd.DataFrame,
    df_comments: pd.DataFrame,
    budget_version: str,
    sample_share: float = 0.8,    # 80% get comments
    status_split=(0.3, 0.7),      # 30% status=1, 70% status=2
    week_anchor: str = "W-MON",   # resample anchor
    seed: int = 42
) -> pd.DataFrame:
    """
    Aggregate Actuals and Budget per week *per BU*, compute Difference, 
    and assign sampled explanations.

    Input:
        df_actuals, df_budget : must have ['date','amount','bu_id']
        df_comments           : must have 'catagory','comment' cols
        sample_share          : fraction of rows to assign comments (~40%)
        status_split          : tuple with probabilities for status 1 and 2
        week_anchor           : e.g. 'W-MON' (ISO weeks ending Monday)
        seed                  : reproducibility

    Output dataframe:
        ['bu_id','date','Year','Week','Actual','Budget','Difference',
         'catagory','comment','status','unexplained_difference']
    """

    rng = np.random.default_rng(seed)

    # --- ensure datetime
    df_actuals = df_actuals.copy()
    df_budget = df_budget.copy()
    df_actuals["date"] = pd.to_datetime(df_actuals["date"], errors="coerce")
    df_budget["date"] = pd.to_datetime(df_budget["date"], errors="coerce")

    # --- aggregate per bu_id + week
    df_actuals_weekly = (
        df_actuals.groupby("bu_id")
        .resample(week_anchor, on="date")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "actual"})
    )

    df_budget_weekly = (
        df_budget.groupby("bu_id")
        .resample(week_anchor, on="date")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "budget"})
    )

    # --- merge & difference
    df = pd.merge(
        df_actuals_weekly, df_budget_weekly,
        on=["bu_id", "date"], how="outer"
    ).fillna(0)

    df["difference"] = df["actual"] - df["budget"]

    # --- ISO week/year
    df["year"] = df["date"].dt.isocalendar().year
    df["week"] = df["date"].dt.isocalendar().week

    # --- init comment/catagory empty
    df["catagory"] = np.nan
    df["comment"] = np.nan

    # --- sample indices (~40%)
    n = int(len(df) * sample_share)
    idx = rng.choice(df.index, size=n, replace=False)

    df_samples = df_comments.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    df.loc[idx, ["catagory", "comment"]] = df_samples[["catagory", "comment"]].values

    # --- assign status
    def _status(row):
        if pd.isna(row["catagory"]) and pd.isna(row["comment"]):
            return 0
        else:
            return rng.choice([1, 2], p=status_split)

    df["status"] = df.apply(_status, axis=1)

    # --- unexplained difference
    df["unexplained_difference"] = np.where(df["status"] == 0, df["difference"], 0)
    df["budget_version"] = budget_version

    def _approved(row):
        if row["status"] == 2:
            return "Approved"
        elif row["status"] == 1:
            return "In Review"
        else:
            return "Unreviewed"

    df["approval_status"] = df.apply(_approved, axis=1)

    return df[[
        "budget_version", "bu_id","date","actual","budget","difference", "unexplained_difference",
        "catagory","comment","status","approval_status"
    ]]
