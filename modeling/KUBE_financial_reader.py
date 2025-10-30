import pandas as pd
from typing import Dict, Any, Union

def _parse_euro_number(x: Union[str, float, int]) -> float:
    """
    Convert values like '28.099.999.744,00' or '-14.000.000,00' to float.
    If it's already numeric, just return float(x).
    """
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    # assume string
    x = x.strip()
    # remove thousand separators '.', swap ',' -> '.'
    x = x.replace(".", "").replace(",", ".")
    try:
        return float(x)
    except ValueError:
        # fallback
        return 0.0
    

def read_financials(path: str, sheet_name: str | int = 0, year: int = 2024) -> Dict[str, Any]:
    """
    Read an Excel file on the form:
    ['Type', 'Account', 'KPMG account', 'Value']
    and return a dict of key financials.

    Returns (example):
    {
        'Revenue': 28099999744.0,
        'COGS': -24710999168.0,
        'GrossProfit': 3388999936.0,
        'FixedCost': -1892000000.0,
        'DepreciationAmortisation': -447000000.0,
        'EBITDA': 1496999936.0,
        'EBIT': 1050000000.0,
        'NetFinancials': -153000000.0,
        'NetTax': -224000000.0,
        'Associates': 65000000.0,
        'NetProfit': 738000000.0,
    }
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df[df["Year"] == year]

    # standardize column names just in case
    df.columns = [c.strip() for c in df.columns]

    # keep only the 4 expected cols (in case file has noise)
    expected_cols = ["Type", "Account", "KPMG account", "Value"]
    df = df[[col for col in df.columns if col in expected_cols]]

    # parse numbers
    df["Value_num"] = df["Value"].apply(_parse_euro_number)

    # helper to sum by KPMG account label
    def sum_kpmg(label: str) -> float:
        mask = df["KPMG account"].astype(str).str.strip().str.lower() == label.lower()
        return float(df.loc[mask, "Value_num"].sum())

    out = {
        "Revenue":                      sum_kpmg("Revenue"),
        "COGS":                         sum_kpmg("COGS"),
        "GrossProfit":                  sum_kpmg("Gross profit"),
        "FixedCost":                    sum_kpmg("Fixed cost"),
        "DepreciationAmortisation":     sum_kpmg("Depreciation & amortisation"),
        "EBITDA":                       sum_kpmg("EBITDA"),
        "EBIT":                         sum_kpmg("EBIT"),
        "NetFinancials":                sum_kpmg("Net financials"),
        "NetTax":                       sum_kpmg("Net tax"),
        "Associates":                   sum_kpmg("Associates, group entities etc."),
        "NetProfit":                    sum_kpmg("Net profit"),
        "AddBackDepreciation":          sum_kpmg("Add-back depreciation & amortisation"),
    }

    # you might also want some derived KPIs:
    # Gross margin %, EBITDA margin, etc.
    rev = out["Revenue"] if out["Revenue"] else 0.0
    out["GrossMarginPct"] = out["GrossProfit"] / rev if rev != 0 else None
    out["EBITDAmarginPct"] = out["EBITDA"] / rev if rev != 0 else None
    out["EBITmarginPct"] = out["EBIT"] / rev if rev != 0 else None
    out["NetMarginPct"] = out["NetProfit"] / rev if rev != 0 else None

    return out


def read_balance_sheet(path: str, sheet_name: str | int = 0, year: int = 2024) -> Dict[str, Any]:
    
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df[df["Year"] == year]

    # normalize cols
    df.columns = [c.strip() for c in df.columns]
    expected_cols = ["Type", "Account", "KPMG account", "Value"]
    df = df[[col for col in df.columns if col in expected_cols]]

    # parse numeric
    df["Value_num"] = df["Value"].apply(_parse_euro_number)

    # helper: sum all rows with given "KPMG account" label (case-insensitive)
    def sum_kpmg(label: str) -> float:
        mask = df["KPMG account"].astype(str).str.strip().str.lower() == label.lower()
        return float(df.loc[mask, "Value_num"].sum())

    out = {
        # Assets side
        "PPE":                              sum_kpmg("PP&E"),
        "FinancialNonCurrentAssets":        sum_kpmg("Financial non-current assets"),
        "IntangibleAssets":                 sum_kpmg("Intangible assets"),
        "TotalNonCurrentAssets":            sum_kpmg("Total non-current assets"),

        "TradeReceivables":                 sum_kpmg("Trade receivables"),
        "TotalReceivables":                 sum_kpmg("Total receivables"),
        "Inventory":                        sum_kpmg("Inventory"),
        "CashAndCashEq":                    sum_kpmg("Cash & cash equivalents"),
        "OtherCurrentAssets":               sum_kpmg("Other current assets"),
        "TotalCurrentAssets":               sum_kpmg("Total current assets"),

        "TotalAssets":                      sum_kpmg("Total assets"),

        # Equity & liabilities side
        "Equity":                           sum_kpmg("Equity"),
        "Provisions":                       sum_kpmg("Provisions"),
        "TotalDebt":                        sum_kpmg("Total debt"),
        "OtherLiabilities":                 sum_kpmg("Other liabilities"),

        "TradePayables":                    sum_kpmg("Trade payables"),
        "OtherPayables": (
            sum_kpmg("Other payables")
        ),

        "TotalLiabilities":                 sum_kpmg("Total liabilities"),
        "TotalEquityAndLiabilities":        sum_kpmg("Total EQ & liabilities"),
    }
    assets = out.get("TotalAssets", 0.0)
    eq_plus_liab = out.get("TotalEquityAndLiabilities", 0.0)
    out["BalanceCheckDiff"] = assets - eq_plus_liab

    return out


def read_company_KUBE(path: str) -> Dict[str, Any]:
    return {
        "pnl": read_financials(path, "Financial statements"),
        "balance": read_balance_sheet(path, "Financial statements"),
    }