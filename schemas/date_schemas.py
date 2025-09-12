import numpy as np
import pandas as pd

# --- helpers ---
def _normalize_pos(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x[x < 0] = 0.0
    s = x.sum()
    if s <= 0:
        # fallback to uniform
        return np.ones_like(x) / len(x)
    return x / s

def _round_counts_to_sum(weights: np.ndarray, total: int) -> np.ndarray:
    """Proportional rounding that preserves the exact total."""
    w = _normalize_pos(np.asarray(weights, dtype=float))
    raw = w * total
    base = np.floor(raw).astype(int)
    remainder = total - base.sum()
    if remainder > 0:
        # assign remaining counts to the largest fractional parts
        frac_idx = np.argsort(-(raw - base))
        base[frac_idx[:remainder]] += 1
    return base

def _business_days_in_month(month_start: pd.Timestamp) -> pd.DatetimeIndex:
    month_end = (month_start + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
    return pd.bdate_range(month_start, month_end)

def _calendar_days_in_month(month_start: pd.Timestamp) -> pd.DatetimeIndex:
    month_end = (month_start + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
    return pd.date_range(month_start, month_end, freq="D")

def _pick_days(days: pd.DatetimeIndex, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(days) == 0 or n <= 0:
        return np.array([], dtype="datetime64[ns]")
    idx = rng.integers(0, len(days), size=n)
    return days.values[idx]

def _month_span_from_dates(dates: np.ndarray) -> pd.PeriodIndex:
    """Return sorted unique months (Period[M]) that actually occur in `dates`."""
    months = pd.to_datetime(dates).to_period("M")
    return pd.PeriodIndex(months.unique()).sort_values()

def schema_constant_balanced(
    df_date: pd.DataFrame,
    amounts: pd.Series,
    *,
    noise_pct: float = 0.0,          # 0 = perfectly equal; set e.g. 0.05 for mild variation
    business_days_only: bool = True,
    random_state: int | None = None,
    balance_mode: str = "abs",        # "abs" or "signed"
):
    """
    Evenly distribute rows and monthly totals across the date span in `df_date`.

    - Ensures each month with target weight receives >=1 row (when n >= #months).
    - Preserves the *grand total* exactly.
    - Equalizes per-month totals (absolute or signed), with optional light noise.
    - Samples days uniformly within each month (no duplicates unless needed).
    """
    rng = np.random.default_rng(random_state)
    n = int(len(amounts))

    all_dates = pd.to_datetime(df_date["date"]).sort_values().values
    if len(all_dates) == 0 or n == 0:
        return pd.DataFrame({"date": pd.to_datetime([]), "amount": []})

    # 1) Unique month grid across the span
    uniq_months = _month_span_from_dates(all_dates)   # PeriodIndex[M]
    month_ts    = uniq_months.to_timestamp()
    m = len(uniq_months)

    # 2) Target month weights (near uniform with optional log-normal noise)
    if noise_pct and noise_pct > 0:
        sigma = np.log1p(noise_pct)
        noise = np.exp(rng.normal(0.0, sigma, size=m))
        span_weights = _normalize_pos(noise)
    else:
        span_weights = np.ones(m, dtype=float) / m

    # 3) Allocate row counts per month (guarantee >=1 per month if n >= m)
    if n >= m:
        base = np.ones(m, dtype=int)           # seed 1 row per month
        remaining = n - m
        more = _round_counts_to_sum(span_weights, remaining)
        month_counts = base + more
    else:
        # fewer rows than months: choose which months get rows by weight
        month_counts = np.zeros(m, dtype=int)
        take_idx = rng.choice(m, size=n, replace=False, p=span_weights)
        for i in take_idx:
            month_counts[i] += 1

    # 4) Pick actual dates within months
    picked = []
    month_index_for_row = []
    for midx, (m_start, cnt) in enumerate(zip(month_ts, month_counts)):
        if cnt <= 0:
            continue
        days = _business_days_in_month(m_start) if business_days_only else _calendar_days_in_month(m_start)
        chosen = _pick_days(days, cnt, rng)
        picked.append(chosen)
        month_index_for_row.extend([midx] * cnt)

    dates = pd.DatetimeIndex(np.concatenate(picked)) if picked else pd.DatetimeIndex([])
    # keep row order stable with dates paired 1:1 to amounts
    df = pd.DataFrame({"date": dates, "amount": amounts.values})
    df["_month"] = pd.PeriodIndex(df["date"].dt.to_period("M"))

    # 5) Compute current per-month totals and targets on *present* months only
    if balance_mode == "abs":
        cur = df.groupby("_month")["amount"].apply(lambda s: s.abs().sum())
        grand = cur.sum()
    else:
        cur = df.groupby("_month")["amount"].sum()
        grand = cur.sum()

    present_months = cur.index  # subset of uniq_months that actually received rows
    pm = len(present_months)

    # Targets: equal per present month (optionally perturbed by small noise), renormalized to exact grand
    if noise_pct and noise_pct > 0:
        sigma = np.log1p(noise_pct)
        n2 = np.exp(rng.normal(0.0, sigma, size=pm))
        w2 = _normalize_pos(n2)
    else:
        w2 = np.ones(pm, dtype=float) / pm

    tgt = pd.Series(grand * w2, index=present_months)

    # 6) Per-month scaling factors (safe for zeros)
    factors = (tgt / cur.replace(0, np.nan)).fillna(0.0)

    # 7) Apply scaling and return (grand total preserved by construction)
    df["amount"] = df["amount"] * df["_month"].map(factors).astype(float)
    return df.drop(columns=["_month"])