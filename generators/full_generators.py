import pandas as pd 
import os
import numpy as np
import utils.utils as utils
import generators.random_generators as random_generators
import generators.llm_generators as llm_generators
import modeling.payroll as payroll
import modeling.erp as erp
import modeling.mapping as mapping
from generators.llm_context_generators import generate_context_numbers_llm, generate_context_report, generate_year_end_report_from_pdf
from typing import Dict, List, Tuple

def create_company_data(company_name: str, save_to_csv: bool = True) -> dict:
    """
    Generates all dimension tables for a company and saves them to CSV files.
    """

    print(f"Generating data for company: {company_name}...")

    path_to_report = f"data/inputdata/reports/generated/{company_name}_context_report.txt"
    if not os.path.exists(path_to_report):
        print(f"No context report found for {company_name}. Generating a new one...")
        generate_context_report(company_name=company_name)
    else:
        print(f"Using existing context report for {company_name}.")
        
    data_context = generate_context_numbers_llm(company_name=company_name)
    
    # Unpack the context data:
    (count_employee, count_product, count_department, count_procurement,
     count_vendor, count_service, count_account, count_customer) = (
        data_context['count_employee'], data_context['count_product'],
        data_context['count_department'], data_context['count_procurement'],
        data_context['count_vendor'], data_context['count_service'],
        data_context['count_account'], data_context['count_customer']
    )

    print(f"Context data for {company_name} generated: {data_context}")

    generated_data = generate_all_dimensions(
        company_name=company_name,
        count_employee=count_employee,
        count_product=count_product,
        count_procurement=count_procurement,
        count_service=count_service,
        count_account=count_account,
        count_customer=count_customer,
        count_vendor=count_vendor,
        count_department=count_department,
        save_to_csv=save_to_csv
    )

    # Generate mapping between all dimensions
    mapped_data = create_mapping_between_all(generated_data=generated_data, company_name=company_name, save_to_csv=save_to_csv)
    # Create all ERP data
    erp_data = create_all_erp_data(generated_mapped_data=mapped_data, company_name=company_name, save_to_csv=save_to_csv)

    print(f"✔ All ERP data and mapping generated for company: {company_name}")

    return {
        "dimensions": generated_data,
        "mapping": mapped_data,
        "erp_data": erp_data
    }

def generate_context_report(company_name: str) -> dict:
    """
    Generate a full context for the given company.
    """

    # Check if report exists; 
    if not os.path.exists(f"data/inputdata/reports/{company_name}.pdf"):
        print(f"The model will produce the data without the year-end report for {company_name} as context...")
        data_report = None
        #data_report = generate_year_end_report_with_web_llm(company_name=company_name)
    else:
        print(f"Loading and reading year-end report for {company_name}.")
        print(f"This usually takes 5-10 mins.")
        data_report = generate_year_end_report_from_pdf(company_name=company_name)
        with open(f"data/inputdata/reports/generated/{company_name}_context_report.txt", "w", encoding="utf-8") as f:
            f.write(str(data_report))

    return data_report


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _validate_length(df: pd.DataFrame, expected: int, name: str) -> None:
    if len(df) != expected:
        raise ValueError(f"Expected {expected} rows in {name}, but got {len(df)}.")

def _even_chunks(total_count: int, max_chunk: int) -> List[int]:
    """
    Split total_count into near-even chunk sizes capped at max_chunk.
    E.g., 370 with max_chunk=150 -> [124, 123, 123] (sums to 370)
    """
    if total_count <= 0:
        return []
    n_chunks = max(1, math.ceil(total_count / max_chunk))
    base = total_count // n_chunks
    rem  = total_count %  n_chunks
    # First 'rem' chunks get +1
    return [base + (1 if i < rem else 0) for i in range(n_chunks)]

def _concat_exact(dfs: List[pd.DataFrame], expected: int, name: str) -> pd.DataFrame:
    """
    Concatenate and then trim or sample to hit the exact expected row count.
    Useful when LLM returns +/- a few rows per chunk.
    """
    if not dfs:
        raise ValueError(f"No dataframes produced for {name}.")
    out = pd.concat(dfs, ignore_index=True)

    if len(out) == expected:
        return out

    if len(out) > expected:
        # Downsample deterministically (but allow seed override upstream if needed)
        # Keep stable subset but preserve variety by sampling w/o replacement
        out = out.sample(n=expected, replace=False, random_state=42).reset_index(drop=True)
        return out

    # If fewer rows than requested, upsample with replacement (last resort)
    need = expected - len(out)
    if len(out) == 0:
        raise ValueError(f"LLM produced 0 rows for {name}, cannot upsample.")
    boost = out.sample(n=need, replace=True, random_state=43)
    out = pd.concat([out, boost], ignore_index=True)
    return out

# ------------------------------------------------------------
# Main generator
# ------------------------------------------------------------
def generate_items_llm(
    company_name: str,
    financial_totals: Dict[str, float],   # e.g., {"Revenue": 1.2e9, "COGS": -8.0e8}
    *,
    df_coa: Optional[pd.DataFrame] = None,
    count_bu: int = 15,
    count_emp: int = 300,
    count_parties: int = 100,
    count_rev: int = 150,
    count_cogs: int = 100,
    # Extra categories, e.g. {"OPEX": 200, "OtherRevenue": 40}
    extra_categories: Optional[Dict[str, int]] = None,
    # LLM knobs
    model: str = "gpt-4.1",
    temp: float = 0.7,
    # Chunking: keep chunks modest so prompts stay fast & stable
    max_chunk_size: int = 150,
    # Saving like your v1
    save_to_csv: bool = True,
    output_dir: str = "data/outputdata_v1"
) -> Dict[str, pd.DataFrame]:
    """
    New unified generator:
    - Generates business units, roles/departments, parties
    - Generates Revenue, COGS (chunked), plus any extra categories (chunked)
    - Returns dict of DataFrames and optionally writes CSVs
    - Enforces exact requested row counts for each generated table
    """

    # ----------------------------
    # 1) Top-level dimensions
    # ----------------------------
    df_bus = llm_generators.generate_bus_llm(company_name=company_name, model=model, temp=temp, count=count_bu)
    _validate_length(df_bus, count_bu, "business_units")

    df_roles, df_departments = llm_generators.generate_roles_and_departments_llm(
        company_name=company_name,
        count=count_emp,
        df_business_units=df_bus
    )
    _validate_length(df_roles, count_emp, "roles")
    _validate_length(df_departments, df_departments.shape[0], "departments")  # keep flexible if LLM expands

    df_parties = llm_generators.generate_parties(
        company_name=company_name, model=model, temp=temp, count=count_parties, df_business_units=df_bus
    )
    _validate_length(df_parties, count_parties, "parties")

    # Optional: mean pay + payroll (parity with your old function)
    # If you don’t need payroll here, you can remove this block.
    mean_pay = llm_generators.estimate_mean_pay_llm(company_name)
    df_employees = utils.sample_employees(count_emp, if_fullname=False)

    df_pay  = payroll.create_pay_roll(df_roles=df_roles, df_employees=df_employees, mean_pay=mean_pay, if_long=False)
    df_payl = payroll.create_pay_roll(df_roles=df_roles, df_employees=df_employees, mean_pay=mean_pay, if_long=True)
    df_pay, df_payroll, df_payline = create_payroll_data(
        df_pay=df_pay, df_payroll=df_payl, df_department=df_departments
    )

    # ----------------------------
    # 2) Line-item categories
    # ----------------------------
    outputs: Dict[str, pd.DataFrame] = {
        "business_units": df_bus,
        "roles": df_roles,
        "departments": df_departments,
        "parties": df_parties,
        "pay": df_pay,
        "payroll": df_payroll,
        "payline": df_payline,
    }

    def _generate_category(
        category_name: str, expected_count: int, total_amount: float
    ) -> pd.DataFrame:
        """
        Chunk, call LLM per chunk, then concat & enforce exact rows.
        Financial total is split proportionally to chunk size, so the
        sum of chunk totals equals the category total.
        """
        if expected_count <= 0:
            return pd.DataFrame()

        chunk_sizes = _even_chunks(expected_count, max_chunk_size)
        dfs = []
        remaining_total = float(total_amount)
        remaining_rows  = int(expected_count)

        for i, chunk_n in enumerate(chunk_sizes):
            # Proportional split of the financial total to chunk rows
            # to avoid odd rounding drift when chunks are unequal.
            if i < len(chunk_sizes) - 1 and remaining_rows > 0:
                frac = chunk_n / remaining_rows
                chunk_total = remaining_total * frac
            else:
                # last chunk: whatever remains
                chunk_total = remaining_total

            df_chunk = generate_line_items_llm(
                company_name=company_name,
                count=chunk_n,
                category_name=category_name,
                financial_total=chunk_total,
                df_business_units=df_bus,
                df_parties=df_parties,
                df_accounts=df_coa
            )
            dfs.append(df_chunk)

            # Update remaining trackers
            remaining_total -= chunk_total
            remaining_rows  -= chunk_n

        df_cat = _concat_exact(dfs, expected_count, name=category_name)
        return df_cat

    # Revenue
    if "Revenue" in financial_totals and count_rev > 0:
        df_revenue = _generate_category(
            "Revenue", count_rev, float(financial_totals["Revenue"])
        )
        outputs["revenue"] = df_revenue

    # COGS
    if "COGS" in financial_totals and count_cogs > 0:
        df_cogs = _generate_category(
            "COGS", count_cogs, float(financial_totals["COGS"])
        )
        outputs["cogs"] = df_cogs

    # Any extra categories (e.g. OPEX, OtherIncome, etc.)
    if extra_categories:
        for cat_name, cat_count in extra_categories.items():
            if cat_count <= 0:
                continue
            total = float(financial_totals.get(cat_name, 0.0))
            df_cat = _generate_category(cat_name, cat_count, total)
            outputs[cat_name.lower()] = df_cat

    df_revenue, df_cogs = utils.mirror_intercompany_flows(df_revenue, df_cogs)
    df_customers = utils.get_party_list(df_revenue, df_parties, df_bus, "customer")
    df_vendors = utils.get_party_list(df_cogs, df_parties, df_bus, "vendor")

    # Function to generate item IDs
    def generate_item_id(prefix="ITEM", start_num=1000):
        """Generate sequential item IDs with a prefix"""
        counter = start_num
        while True:
            yield f"{prefix}{counter:04d}"
            counter += 1

    # Create item IDs for products and COGS
    df_products = df_revenue[["item_name", "unit_price"]].drop_duplicates("item_name")
    df_spend = df_cogs[["item_name", "unit_price"]].drop_duplicates("item_name")

    # Generate item IDs for products
    item_id_gen = generate_item_id("", 1000)
    df_products["item_id"] = [next(item_id_gen) for _ in range(len(df_products))]

    # Generate item IDs for COGS items  
    item_id_gen_cogs = generate_item_id("", 2000)
    df_spend["item_id"] = [next(item_id_gen_cogs) for _ in range(len(df_spend))]

    if save_to_csv:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/dimensions", exist_ok=True)
        os.makedirs(f"{output_dir}/fact", exist_ok=True)

        # Dimensions
        outputs["business_units"].to_csv(f"{output_dir}/dimensions/business_unit.csv", index=False)
        outputs["roles"].to_csv(f"{output_dir}/dimensions/role.csv", index=False)
        outputs["departments"].to_csv(f"{output_dir}/dimensions/department.csv", index=False)
        df_customers.to_csv(f"{output_dir}/dimensions/customer.csv", index=False)
        df_vendors.to_csv(f"{output_dir}/dimensions/vendors.csv", index=False)
        df_items.to_csv(f"{output_dir}/dimensions/cogs_items.csv", index=False)
        df_products.to_csv(f"{output_dir}/dimensions/products.csv", index=False)
        
        # Facts
        outputs["payroll"].to_csv(f"{output_dir}/fact/erp_payroll.csv", index=False)
        # Extra categories go to /fact/<cat>.csv
        for k, v in outputs.items():
            if k not in {"business_units","roles","departments","parties","pay","payroll","payline","revenue","cogs"}:
                if isinstance(v, pd.DataFrame) and not v.empty:
                    v.to_csv(f"{output_dir}/fact/{k}_lines.csv", index=False)
    
    return outputs


def generate_all_dimensions_v1(company_name: str,
                            report: str = None,
                            count_employee: int = 200,
                            count_product: int = 100,
                            count_account: int = 30,
                            count_customer: int = 30,
                            count_department: int = 20,
                            count_service: int = 100,
                            count_procurement: int = 100,
                            save_to_csv: bool = True) -> dict:

    print(f"Generating data for company: {company_name}...")

    def validate_length(df, expected, name):
        if len(df) != expected:
            raise ValueError(f"Expected {expected} rows in {name}, but got {len(df)}.")

    # Generate employees and roles
    df_roles = llm_generators.generate_roles_llm(company_name, count_employee)
    df_employees = utils.sample_employees(count_employee, if_fullname=False)
    validate_length(df_roles, count_employee, "roles")
    validate_length(df_employees, count_employee, "names")
    print("✔ Roles and Names generated.")

    # Generate products and services
    df_procurement = llm_generators.generate_procurement_llm(company_name, count_procurement)
    validate_length(df_procurement, count_procurement, "procurement")
    print("✔ Procurement data generated.")
    df_service = llm_generators.generate_services_llm(company_name, count_service)
    validate_length(df_service, count_service, "service")
    print("✔ Services data generated.")
    df_product = llm_generators.generate_sales_products_llm(company_name, count_product)
    validate_length(df_product, count_product, "product")
    print("✔ Products data generated.")

    # Generate other dimension tables
    df_account    = llm_generators.generate_accounts_llm(company_name, count_account)
    df_customer   = llm_generators.generate_customers_llm(company_name, count_customer)
    df_department = llm_generators.generate_departments_llm(company_name, count_department)
    df_vendor     = llm_generators.generate_vendors_llm(company_name, count_customer)
    validate_length(df_account, count_account, "account")
    validate_length(df_customer, count_customer, "customer")
    validate_length(df_department, count_department, "department_name")
    validate_length(df_vendor, count_customer, "vendor")
    print("✔ Accounts, Customers, Departments, and Vendors generated.")

    # Convert proportionality
    dfs = [df_procurement, df_service, df_product, df_customer, df_department]
    dfs = [utils.convert_column_to_percentage(df, 'proportionality', scale=1.0) for df in dfs]

    # Then unpack if needed
    df_procurement, df_service, df_product, df_customer, df_department = dfs


    mean_pay = llm_generators.estimate_mean_pay_llm(company_name)

    df_pay = payroll.create_pay_roll(
        df_roles = df_roles,
        df_employees = df_employees,
        mean_pay = mean_pay,
        if_long=False  # Set to True for long format
    )

    df_payroll = payroll.create_pay_roll(
        df_roles = df_roles,
        df_employees = df_employees,
        mean_pay = mean_pay,
        if_long=True  # Set to True for long format
    )
    
    df_pay, df_payroll, df_linemap = create_payroll_data(df_pay=df_pay, df_payroll=df_payroll, df_department=df_department)
    df_linemap = df_linemap.drop_duplicates(subset='name')
    print("✔ Payroll data generated.")

    # Save files if requested
    if save_to_csv:
        output_dir = f"data/outputdata/"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/dimensions", exist_ok=True)
        os.makedirs(f"{output_dir}/fact", exist_ok=True)

        df_pay.to_csv(f"{output_dir}/dimensions/employee.csv", index=False)
        df_payroll.to_csv(f"{output_dir}/fact/erp_payroll.csv", index=False)
        df_linemap.to_csv(f"{output_dir}/dimensions/payline.csv", index=False)
        df_procurement.to_csv(f"{output_dir}/dimensions/procurement.csv", index=False)
        df_service.to_csv(f"{output_dir}/dimensions/service.csv", index=False)
        df_product.to_csv(f"{output_dir}/dimensions/product.csv", index=False)
        df_account.to_csv(f"{output_dir}/dimensions/account.csv", index=False)
        df_customer.to_csv(f"{output_dir}/dimensions/customer.csv", index=False)
        df_department.to_csv(f"{output_dir}/dimensions/department.csv", index=False)
        df_vendor.to_csv(f"{output_dir}/dimensions/vendor.csv", index=False)

        print(f"✔ All CSVs saved to: {output_dir}")

    return {
        "pay": df_pay,
        "payroll": df_payroll,
        "payline": df_linemap,
        "procurement": df_procurement,
        "service": df_service,
        "product": df_product,
        "account": df_account,
        "customer": df_customer,
        "department_name": df_department,
        "vendor": df_vendor
    }

def create_payroll_data(df_pay: pd.DataFrame, df_payroll: pd.DataFrame, df_department: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    
    """
    df_erp_payroll = payroll.add_taxes(df_payroll=df_payroll)
    df_map_payroll = pd.read_csv("data/inputdata/line_id_accounts.csv")

    df_pay = utils.assign_departments(df_pay, df_department)
    df_erp_payroll_full, df_line_mapping = payroll.create_full_payroll(df_payroll=df_erp_payroll, df_mapping=df_map_payroll)
    return df_pay, df_erp_payroll_full, df_line_mapping

def create_mapping_between_all(generated_data: dict = None, company_name: str = None, save_to_csv: bool = True) -> dict:
    """
    Creates a mapping between products, services, payroll, and GL accounts,
    using either provided generated data or loaded CSVs from a folder.

    Args:
        generated_data (dict, optional): Dictionary containing all generated dimension tables.
        company_name (str, optional): If provided, will load data from CSVs located at
                                      data/outputdata/

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_spend, df_mapped)
    """

    if not generated_data and not company_name:
        raise ValueError("You must provide either 'generated_data' or 'company_name'.")

    if company_name:
        base_path = f"data/outputdata/dimensions/"
        try:
            df_products    = pd.read_csv(os.path.join(base_path, "product.csv"))
            df_services    = pd.read_csv(os.path.join(base_path, "service.csv"))
            df_procurement = pd.read_csv(os.path.join(base_path, "procurement.csv"))
            df_departments = pd.read_csv(os.path.join(base_path, "department.csv"))
            df_accounts    = pd.read_csv(os.path.join(base_path, "account.csv"))
            df_customers   = pd.read_csv(os.path.join(base_path, "customer.csv"))
            df_vendors     = pd.read_csv(os.path.join(base_path, "vendor.csv"))
            df_payroll     = pd.read_csv(os.path.join("data/outputdata/fact/", "erp_payroll.csv"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing file in folder: {e.filename}")
    else:
        df_products    = generated_data["product"].copy()
        df_services    = generated_data["service"].copy()
        df_procurement = generated_data["procurement"].copy()
        df_departments = generated_data["department_name"].copy()
        df_accounts    = generated_data["account"].copy()
        df_customers   = generated_data["customer"].copy()
        df_payroll     = generated_data["payroll"].copy()
        df_vendors     = generated_data["vendor"].copy()

    estimated_financials = erp.estimate_costs_from_payroll(df_pay=df_payroll)


    # Apply proportional spend estimates
    df_procurement["annual_spend"] = np.round(
        estimated_financials["estimated_product"] * df_procurement["proportionality"], -3
    )
    df_products["annual_spend"] = np.round(
        estimated_financials["estimated_revenue"] * df_products["proportionality"], -3
    )
    df_services["annual_spend"] = np.round(
        estimated_financials["estimated_service"] * df_services["proportionality"], -3
    )
    df_departments["annual_spend"] = np.round(
        estimated_financials["estimated_payroll"] * df_departments["proportionality"], -3
    )
    print("\n * Semantic mapping started * :")
    print("Time estimate: 5-7 minutes")
    df_erp_expenses, df_map_expenses = mapping.map_procurement_services(df_procurement=df_procurement, df_services=df_services, df_accounts=df_accounts, df_departments=df_departments, df_customers=df_customers, df_vendors=df_vendors)
    df_erp_products, df_map_products = mapping.map_products(df_products=df_products, df_accounts=df_accounts, df_departments=df_departments, df_customers=df_customers, df_vendors=df_vendors)

    print(f"✔ All mapping data generated.")

    if save_to_csv:
        output_dir = f"data/outputdata/mapping"
        os.makedirs(output_dir, exist_ok=True)
        df_map_expenses.to_csv(f"{output_dir}/map_expenses.csv", index=False)
        df_map_products.to_csv(f"{output_dir}/map_products.csv", index=False)
        print(f"✔ All mapping CSVs saved to: {output_dir}")

    return {
        "df_erp_expenses": df_erp_expenses,
        "df_map_expenses": df_map_expenses,
        "df_erp_products": df_erp_products,
        "df_map_products": df_map_products,
    }


def create_all_erp_data(generated_mapped_data: dict, company_name: str, save_to_csv: bool = True) -> dict:
    """
    Creates all ERP-related data by combining generated and mapped data.
    """

    df_map_products = generated_mapped_data["df_map_products"]
    df_map_expenses = generated_mapped_data["df_map_expenses"]
    df_erp_products = generated_mapped_data["df_erp_products"]
    df_erp_expenses = generated_mapped_data["df_erp_expenses"]

    document_metadata_expense = random_generators.generate_document_metadata(n=30, start_index=1000)
    document_metadata_products = random_generators.generate_document_metadata(n=30, start_index=2000)
    
    df_erp_expenses_full = erp.create_erp_data(df_expenses=df_erp_expenses, df_expenses_mapping=df_map_expenses, df_document_metadata=document_metadata_expense)
    df_erp_products_full = erp.create_erp_data(df_expenses=df_erp_products, df_expenses_mapping=df_map_products, df_document_metadata=document_metadata_products)
      
    # Full target schema # also currency, amount_eur, Type
    full_columns = ['document_number', 'type', 'date', 'amount_dkk', 'account_name', 'product_id', 'procurement_id', 'service_id']
    vendor_col = ['vendor_name']    
    customer_col = ['customer_name']

    # Reindex all ERP dataframes to align to full schema
    df_expenses_full = df_erp_expenses_full.reindex(columns=full_columns + vendor_col)
    df_products_full = df_erp_products_full.reindex(columns=full_columns + customer_col)


    rename_cols = {
        'document_number': 'document_number',
        'type': 'debit_credit', 
        'date': 'date',
        'amount_dkk': 'amount',
        'account_name': 'account_id',
        'product_id': 'product_id',
        'procurement_id': 'procurement_id',
        'service_id': 'service_id',
        'vendor_name': 'vendor_id',
        'customer_name': 'customer_id'}
    


    df_expenses_full.rename(columns=rename_cols, inplace=True)
    df_products_full.rename(columns=rename_cols, inplace=True)
    
    if save_to_csv:
        output_dir = f"data/outputdata/fact"
        os.makedirs(output_dir, exist_ok=True)
        df_erp_expenses_full.to_csv(f"{output_dir}/erp_expenses.csv", index=False)
        df_erp_products_full.to_csv(f"{output_dir}/erp_products.csv", index=False)


    # Concatenate all ERP data
    df_erp_all = pd.concat([df_expenses_full, df_products_full], ignore_index=True)
    df_accounts = pd.read_csv("data/outputdata/dimensions/account.csv")

    df_erp_all = erp.balance_documents_with_assets(df_erp=df_erp_all, df_accounts=df_accounts, tolerance=100)
    
    print(f"✔ All erp-data generated.")

    if save_to_csv:
        output_dir = f"data/outputdata/fact"
        os.makedirs(output_dir, exist_ok=True)
        df_erp_expenses_full.to_csv(f"{output_dir}/erp_expenses.csv", index=False)
        df_erp_products_full.to_csv(f"{output_dir}/erp_products.csv", index=False)
        df_erp_all.to_csv(f"{output_dir}/general_ledger.csv", index=False)
        print(f"✔ All ERP CSVs saved to: {output_dir}")
    
    return {
        "df_erp_expenses_full": df_erp_expenses_full,
        "df_erp_products_full": df_erp_products_full,
        "df_erp_all": df_erp_all,
    }