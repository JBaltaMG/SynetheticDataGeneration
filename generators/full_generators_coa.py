from multiprocessing import context
import pandas as pd 
import os
import numpy as np
import json
from pathlib import Path
from modeling.erp_fast import balance_monthly, mirror_intercompany_expenses_to_products
from schemas.date_schemas import schema_constant_balanced
import utils.utils as utils
import utils.utils_llm as utils_llm
import generators.random_generators as random_generators
import generators.llm_generators as llm_generators
import modeling.payroll as payroll
import modeling.erp as erp
import modeling.mapping as mapping
import modeling.budget as budget
from generators.llm_context_generators import generate_context_numbers_llm, generate_context_report, generate_year_end_report_from_pdf
from typing import Dict, List, Tuple

def read_cached_context_report(company_name: str) -> dict:
    """
    Read cached context report from JSON file.
    Returns the parsed JSON dict or None on missing/parse error.
    """
    try:
        context_path = Path(f"data/inputdata/reports/generated/{company_name}_context.json")
        if context_path.exists():
            with open(context_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return None

def create_company_data(company_name: str, save_to_csv: bool = True, context: bool = False) -> dict:
    """
    Generates all dimension tables for a company and saves them to CSV files.
    """

    print(f"Generating data for company: {company_name}...")


    if context:
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

def generate_all_dimensions(
    company_name: str,
    count_employee: int = 200,
    count_product: int = 100,
    count_account: int = 30,
    count_customer: int = 30,
    count_department: int = 20,
    count_service: int = 100,
    count_procurement: int = 100,
    count_vendor: int = 30,
    save_to_csv: bool = True,
    max_retries: int = 3,             
) -> dict:
    """
    Generates all dimensions, validates final lengths, and retries if any mismatch.
    CSVs are saved only after a fully valid generation.
    """
    print(f"Generating dimensions for company: {company_name}.")
    print(f"This usually takes 2-5 mins.")
    def _run_once() -> dict:

        # Employees / Roles
        df_roles = llm_generators.generate_roles_llm(company_name, count_employee)
        df_employees = utils.sample_employees(count_employee, if_fullname=False)
        print("✔ Roles and Names generated.")

        # Procurement / Services / Products
        df_procurement = llm_generators.generate_procurement_llm(company_name, count_procurement)
        print("✔ Procurement data generated.")
        df_service = llm_generators.generate_services_llm(company_name, count_service)
        print("✔ Services data generated.")
        df_product = llm_generators.generate_sales_products_llm(company_name, count_product)
        print("✔ Products data generated.")

        # Accounts / Customers / Departments / Vendors
        #df_account    = llm_generators.generate_accounts_llm(company_name, count_account)
        df_account     = pd.read_csv("data/inputdata/coa_general.csv", sep=";")
        #print("✔ Accounts data generated.")
        df_customer   = llm_generators.generate_customers_llm(company_name, count_customer)
        print("✔ Customers data generated.")
        df_department = llm_generators.generate_departments_llm(company_name, count_department)
        print("✔ Departments data generated.")
        df_vendor     = llm_generators.generate_vendors_llm(company_name, count_vendor)
        print("✔ Vendors data generated.")

        df_bu_companies = llm_generators.generate_business_units_llm(company_name, 9)
        print("✔ Business Units and company data generated.")

        # If your LLM functions already convert 'proportionality', you can remove this block.
        # Keeping as-is to match your current logic.
        dfs_to_pct = [df_procurement, df_service, df_product, df_customer, df_department]
        dfs_to_pct = [utils.convert_column_to_percentage(df, 'proportionality', scale=1.0) for df in dfs_to_pct]
        df_procurement, df_service, df_product, df_customer, df_department = dfs_to_pct

        # Payroll
        mean_pay = llm_generators.estimate_mean_pay_llm(company_name)
        df_pay = payroll.create_pay_roll(
            df_roles=df_roles,
            df_employees=df_employees,
            mean_pay=mean_pay,
            if_long=False,
        )
        df_payroll = payroll.create_pay_roll(
            df_roles=df_roles,
            df_employees=df_employees,
            mean_pay=mean_pay,
            if_long=True,
        )
        df_pay, df_payroll, df_linemap = create_payroll_data(
            df_pay=df_pay, df_payroll=df_payroll, df_department=df_department
        )
        df_linemap = df_linemap.drop_duplicates(subset='name')
        print("✔ Payroll data generated.")

        return {
            "roles": df_roles,
            "names": df_employees,
            "procurement": df_procurement,
            "service": df_service,
            "product": df_product,
            "account": df_account,
            "customer": df_customer,
            "department_name": df_department,
            "vendor": df_vendor,
            "pay": df_pay,
            "payroll": df_payroll,
            "payline": df_linemap,
            "bu": df_bu_companies,
        }

    def _validate_all(res: dict) -> Tuple[bool, List[str]]:
        """Validate final lengths at the end; return (ok, list_of_errors)."""
        errors = []

        expected = {
            "roles":            count_employee,
            "names":            count_employee,
            "procurement":      count_procurement,
            "service":          count_service,
            "product":          count_product,
            #"account":          count_account,
            "customer":         count_customer,
            "department_name":  count_department,
            "vendor":           count_vendor,
            "bu":     9,
        }

        for key, exp in expected.items():
            got = len(res[key])
            if got != exp:
                errors.append(f"{key}: expected {exp}, got {got}")

        return (len(errors) == 0, errors)

    # --- Retry loop ---
    last_errors: List[str] = []
    for attempt in range(1, max_retries + 1):
        print(f"\n=== Attempt {attempt} ===")
        try:
            result = _run_once()
            ok, errors = _validate_all(result)
            if ok:
                print("✔ All table lengths validated.")

                if save_to_csv:
                    output_dir = "data/outputdata"
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(f"{output_dir}/dimensions", exist_ok=True)
                    os.makedirs(f"{output_dir}/fact", exist_ok=True)

                    result["pay"].to_csv(f"{output_dir}/dimensions/employee.csv", index=False)
                    result["payroll"].to_csv(f"{output_dir}/fact/erp_payroll.csv", index=False)
                    result["payline"].to_csv(f"{output_dir}/dimensions/payline.csv", index=False)
                    result["procurement"].to_csv(f"{output_dir}/dimensions/procurement.csv", index=False)
                    result["service"].to_csv(f"{output_dir}/dimensions/service.csv", index=False)
                    result["product"].to_csv(f"{output_dir}/dimensions/product.csv", index=False)
                    result["account"].to_csv(f"{output_dir}/dimensions/account.csv", index=False)
                    result["customer"].to_csv(f"{output_dir}/dimensions/customer.csv", index=False)
                    result["department_name"].to_csv(f"{output_dir}/dimensions/department.csv", index=False)
                    result["vendor"].to_csv(f"{output_dir}/dimensions/vendor.csv", index=False)
                    result["bu"].to_csv(f"{output_dir}/dimensions/bu.csv", index=False)
                    print(f"✔ All CSVs saved to: {output_dir}")

                return result

            # Not OK → log and retry
            print("✖ Length validation failed:")
            for e in errors:
                print("  -", e)
            last_errors = errors

        except Exception as ex:
            # Hard failure; collect and retry
            msg = f"Runtime error: {ex}"
            print("✖ Generation failed:", msg)
            last_errors = [msg]

    # If we get here, all attempts failed
    raise ValueError(
        "Generation failed after retries. Last validation errors:\n  - " + "\n  - ".join(last_errors)
    )


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
        df_bu_companies.to_csv(f"{output_dir}/dimensions/bu.csv", index=False)

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
            df_products     = pd.read_csv(os.path.join(base_path, "product.csv"))
            df_services     = pd.read_csv(os.path.join(base_path, "service.csv"))
            df_procurement  = pd.read_csv(os.path.join(base_path, "procurement.csv"))
            df_departments  = pd.read_csv(os.path.join(base_path, "department.csv"))
            df_accounts     = pd.read_csv(os.path.join(base_path, "account.csv"))
            df_customers    = pd.read_csv(os.path.join(base_path, "customer.csv"))
            df_vendors      = pd.read_csv(os.path.join(base_path, "vendor.csv"))
            df_bu_companies = pd.read_csv(os.path.join(base_path, "bu.csv"))
            df_payroll      = pd.read_csv(os.path.join("data/outputdata/fact/", "erp_payroll.csv"))
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
    print("Time estimate: 3-5 minutes")
    df_erp_expenses, df_map_expenses = mapping.map_procurement_services(df_procurement=df_procurement, df_services=df_services, df_accounts=df_accounts, df_customers=df_customers, df_bu_companies=df_bu_companies, df_vendors=df_vendors)
    df_erp_products, df_map_products = mapping.map_products(df_products=df_products, df_accounts=df_accounts, df_customers=df_customers, df_bu_companies=df_bu_companies, df_vendors=df_vendors)

    # Pick random indices once (shared between products & expenses)
    
    df_map_products, df_map_expenses = mapping.pick_intercomp(df_map_products=df_map_products, df_map_expenses=df_map_expenses, df_bu_companies=df_bu_companies)
    
    df_customers, df_vendors, df_bu_companies = mapping.remap_vendors_customers_with_bu(df_customers=df_customers, df_vendors=df_vendors, df_bu_companies=df_bu_companies)

    df_bu_companies.rename(columns={"bu_key": "name"}, inplace=True)

    print(f"✔ All mapping data generated.")

    if save_to_csv:
        df_customers.to_csv(f"data/outputdata/dimensions/customer.csv", index=False)
        df_vendors.to_csv(f"data/outputdata/dimensions/vendor.csv", index=False)
        df_bu_companies.to_csv(f"data/outputdata/dimensions/bu.csv", index=False)

        output_dir = f"data/outputdata/mapping"
        os.makedirs(output_dir, exist_ok=True)
        df_erp_expenses.to_csv(f"{output_dir}/erp_expenses.csv", index=False)
        df_erp_products.to_csv(f"{output_dir}/erp_products.csv", index=False)
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
    from modeling.erp_fast import balance_monthly, create_erp_data_fast, mirror_intercompany_expenses_to_products
    

    document_metadata_expense = random_generators.generate_document_metadata(n=200, start_index=1000)
    document_metadata_products = random_generators.generate_document_metadata(n=400, start_index=2000)

    df_map_products = generated_mapped_data["df_map_products"]
    df_map_expenses = generated_mapped_data["df_map_expenses"]
    df_erp_products = generated_mapped_data["df_erp_products"]
    df_erp_expenses = generated_mapped_data["df_erp_expenses"]

    # --- expenses ERP data (patched pipeline handles duplicates + passthroughs) ---
    data_expenses = create_erp_data_fast(
        df_expenses=df_erp_expenses,
        df_mapping=df_map_expenses,
        df_document_metadata=document_metadata_expense,
        balance_documents=True,      # if you want every doc to sum to 0
        balance_tolerance=100.0,     # how close to 0 before correction lines added
        year_start=2024,
        year_end=2025,
        target_qty_per_line=20.0,
        qty_sigma=0.5,
        lines_for_band=(20, 4),
        seed=42
    )

    # --- products ERP data (same call, just with product mapping/metadata) ---
    data_products = create_erp_data_fast(
        df_expenses=df_erp_products,
        df_mapping=df_map_products,
        df_document_metadata=document_metadata_products,
        balance_documents=True,
        balance_tolerance=100.0,
        year_start=2024,
        year_end=2025,
        target_qty_per_line=5.0,
        qty_sigma=0.5,
        lines_for_band=(12, 2),
        seed=43
    )

    data_products = mirror_intercompany_expenses_to_products(
       df_products=data_products,
       df_expenses=data_expenses,
    )
    
    
    df_expenses_full = balance_monthly(data_expenses, target_types=("procurement", "services"), noise_pct=0.01)

    df_products_full = balance_monthly(data_products, target_types=("Product Sales",), noise_pct=0.01) 
    

    full_columns = ['document_number', 'type', 'date', 'amount', 'quantity', 'account_name', 'product_id', 'procurement_id', 'service_id', 'bu_id']
    vendor_col = ['vendor_name']
    customer_col = ['customer_name']

    # Reindex all ERP dataframes to align to full schema
    df_expenses_full = df_expenses_full.reindex(columns=full_columns + vendor_col)
    df_products_full = df_products_full.reindex(columns=full_columns + customer_col)

    
    rename_cols = {
        'document_number': 'document_number',
        'type': 'debit_credit', 
        'date': 'date',
        'amount': 'amount',
        'quantity': 'quantity',
        'account_name': 'account_id',
        'bu_id': 'bu_id',
        'product_id': 'product_id',
        'procurement_id': 'procurement_id',
        'service_id': 'service_id',
        'vendor_name': 'vendor_id',
        'customer_name': 'customer_id'}
            
    df_expenses_full.rename(columns=rename_cols, inplace=True)
    df_products_full.rename(columns=rename_cols, inplace=True)

    df_erp_all = pd.concat([df_expenses_full, df_products_full], axis=0, ignore_index=True)

    df_erp_budget = budget.generate_budget_from_gl_all_years(df_erp_all)

    df_erp_all["version"] = "Actual"
    df_erp_budget["version"] = "Budget"

    df = pd.concat([df_erp_all, df_erp_budget], axis=0, ignore_index=True)

    print(f"✔ All erp-data generated.")

    if save_to_csv:
        output_dir = f"data/outputdata/fact"
        os.makedirs(output_dir, exist_ok=True)
        df_expenses_full.to_csv(f"{output_dir}/erp_expenses.csv", index=False)
        df_products_full.to_csv(f"{output_dir}/erp_products.csv", index=False)
        df.to_csv(f"{output_dir}/general_ledger.csv", index=False)
        #df_erp_budget.to_csv(f"{output_dir}/fact_budget.csv", index=False)
        print(f"✔ All ERP CSVs saved to: {output_dir}")
    
    return {
        "df_erp_expenses_full": df_expenses_full,
        "df_erp_products_full": df_products_full,
        "df_erp_all": df,
        #"df_erp_budget": df_erp_budget
    }