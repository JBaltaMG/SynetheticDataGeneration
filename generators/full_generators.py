import pandas as pd 
import os
import numpy as np
import utils.utils as utils
import generators.random_generators as random_generators
import generators.llm_generators as llm_generators
import modeling.payroll as payroll
import modeling.erp as erp
from tqdm import tqdm


def create_company_data(company_name: str, count_employee: int = 100, count_products: int = 50,
                        count_accounts: int = 30, count_customers: int = 10,
                        count_departments: int = 10, save_to_csv: bool = True) -> dict:
    """
    Generates all dimension tables for a company and saves them to CSV files.
    """
    generated_data = generate_all_dimensions(
        company_name=company_name,
        count_employee=count_employee,
        count_products=count_products,
        count_accounts=count_accounts,
        count_customers=count_customers,
        count_departments=count_departments,
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


def generate_all_dimensions(company_name: str,
                            count_employee: int = 100,
                            count_products: int = 50,
                            count_accounts: int = 30,
                            count_customers: int = 10,
                            count_departments: int = 10,
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
    print("✔ Payroll data generated.")

    # Generate products and services
    df_procurement = llm_generators.generate_procurement_llm(company_name, count_products)
    validate_length(df_procurement, count_products, "procurement")
    print("✔ Procurement data generated.")
    df_services = llm_generators.generate_services_llm(company_name, count_products)
    validate_length(df_services, count_products, "services")
    print("✔ Services data generated.")
    df_products = llm_generators.generate_sales_products_llm(company_name, count_products)
    validate_length(df_products, count_products, "products")
    print("✔ Products data generated.")

    # Generate other dimension tables
    df_accounts    = llm_generators.generate_accounts_llm(company_name, count_accounts)
    df_customers   = llm_generators.generate_customers_llm(company_name, count_customers)
    df_departments = llm_generators.generate_departments_llm(company_name, count_departments)
    validate_length(df_accounts, count_accounts, "accounts")
    validate_length(df_customers, count_customers, "customers")
    validate_length(df_departments, count_departments, "departments")
    print("✔ Accounts, Customers, and Departments generated.")

    # Convert proportionality
    for df, _ in zip([df_procurement, df_services, df_products, df_customers, df_departments],
                         ["procurement", "services", "products", "customers", "departments"]):
        df = utils.convert_column_to_percentage(df, 'Proportionality', scale=100.0)

    # Save files if requested
    if save_to_csv:
        output_dir = f"data/outputdata/{company_name}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/dimensions", exist_ok=True)

        df_pay.to_csv(f"{output_dir}/dimensions/pay.csv", index=False)
        df_payroll.to_csv(f"{output_dir}/dimensions/payroll.csv", index=False)
        df_procurement.to_csv(f"{output_dir}/dimensions/procurement.csv", index=False)
        df_services.to_csv(f"{output_dir}/dimensions/services.csv", index=False)
        df_products.to_csv(f"{output_dir}/dimensions/products.csv", index=False)
        df_accounts.to_csv(f"{output_dir}/dimensions/accounts.csv", index=False)
        df_customers.to_csv(f"{output_dir}/dimensions/customers.csv", index=False)
        df_departments.to_csv(f"{output_dir}/dimensions/departments.csv", index=False)
        print(f"✔ All CSVs saved to: {output_dir}")

    return {
        "pay": df_pay,
        "payroll": df_payroll,
        "procurement": df_procurement,
        "services": df_services,
        "products": df_products,
        "accounts": df_accounts,
        "customers": df_customers,
        "departments": df_departments
    }

def create_mapping_between_all(generated_data: dict = None, company_name: str = None, save_to_csv: bool = True) -> dict:
    """
    Creates a mapping between products, services, payroll, and GL accounts,
    using either provided generated data or loaded CSVs from a folder.

    Args:
        generated_data (dict, optional): Dictionary containing all generated dimension tables.
        company_name (str, optional): If provided, will load data from CSVs located at
                                      data/outputdata/{company_name}_data/

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_spend, df_mapped)
    """

    if not generated_data and not company_name:
        raise ValueError("You must provide either 'generated_data' or 'company_name'.")

    if company_name:
        base_path = f"data/outputdata/{company_name}/dimensions/"
        try:
            df_products    = pd.read_csv(os.path.join(base_path, "products.csv"))
            df_services    = pd.read_csv(os.path.join(base_path, "services.csv"))
            df_procurement = pd.read_csv(os.path.join(base_path, "procurement.csv"))
            df_departments = pd.read_csv(os.path.join(base_path, "departments.csv"))
            df_accounts    = pd.read_csv(os.path.join(base_path, "accounts.csv"))
            df_customers   = pd.read_csv(os.path.join(base_path, "customers.csv"))
            df_payroll     = pd.read_csv(os.path.join(base_path, "payroll.csv"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing file in folder: {e.filename}")
    else:
        df_products    = generated_data["products"].copy()
        df_services    = generated_data["services"].copy()
        df_procurement = generated_data["procurement"].copy()
        df_departments = generated_data["departments"].copy()
        df_accounts    = generated_data["accounts"].copy()
        df_customers   = generated_data["customers"].copy()
        df_payroll     = generated_data["payroll"].copy()

    estimated_financials = erp.estimate_costs_from_payroll(df_pay=df_payroll)

    # Apply proportional spend estimates
    df_procurement["TotalAnnualSpend"] = np.round(
        estimated_financials["EstimatedProductCost"] * df_procurement["Proportionality"], -3
    )
    df_products["TotalAnnualSpend"] = np.round(
        estimated_financials["EstimatedProductCost"] * df_products["Proportionality"], -3
    )
    df_services["TotalAnnualSpend"] = np.round(
        estimated_financials["EstimatedServiceCost"] * df_services["Proportionality"], -3
    )
    df_departments["TotalAnnualSpend"] = np.round(
        estimated_financials["EstimatedPayroll"] * df_departments["Proportionality"], -3
    )

    df_erp_expenses, df_map_expenses = utils.map_procurement_services(df_procurement=df_procurement, df_services=df_services, df_accounts=df_accounts, df_departments=df_departments, df_customers=df_customers)
    df_erp_products, df_map_products = utils.map_products(df_products=df_products, df_accounts=df_accounts, df_departments=df_departments, df_customers=df_customers)

    df_erp_payroll = payroll.add_taxes(df_payroll=df_payroll)
    df_map_payroll = pd.read_csv("data/inputdata/line_id_accounts.csv")

    print(f"✔ All mapping data generated.")

    if save_to_csv:
        output_dir = f"data/outputdata/{company_name}/mapping"
        os.makedirs(output_dir, exist_ok=True)
        df_map_expenses.to_csv(f"{output_dir}/map_expenses.csv", index=False)
        df_map_payroll.to_csv(f"{output_dir}/map_payroll.csv", index=False)
        df_map_products.to_csv(f"{output_dir}/map_products.csv", index=False)
        print(f"✔ All mapping CSVs saved to: {output_dir}")

    return {
        "df_erp_expenses": df_erp_expenses,
        "df_map_expenses": df_map_expenses,
        "df_erp_payroll": df_erp_payroll,
        "df_map_payroll": df_map_payroll,
        "df_erp_products": df_erp_products,
        "df_map_products": df_map_products,
    }


def create_all_erp_data(generated_mapped_data: dict, company_name: str, save_to_csv: bool = True) -> dict:
    """
    Creates all ERP-related data by combining generated and mapped data.
    """

    df_map_payroll = generated_mapped_data["df_map_payroll"]
    df_map_products = generated_mapped_data["df_map_products"]
    df_map_expenses = generated_mapped_data["df_map_expenses"]
    df_erp_payroll = generated_mapped_data["df_erp_payroll"]
    df_erp_products = generated_mapped_data["df_erp_products"]
    df_erp_expenses = generated_mapped_data["df_erp_expenses"]

    document_metadata_expense = random_generators.generate_document_metadata(n=30, start_index=1000)
    document_metadata_products = random_generators.generate_document_metadata(n=30, start_index=2000)
    #document_metadata_payroll = random_generators.generate_document_metadata(n=30, start_index=3000)

    df_erp_expenses_full = erp.create_erp_data(df_expenses=df_erp_expenses, df_expenses_mapping=df_map_expenses, df_document_metadata=document_metadata_expense)
    df_erp_products_full = erp.create_erp_data(df_expenses=df_erp_products, df_expenses_mapping=df_map_products, df_document_metadata=document_metadata_products)
    df_erp_payroll_full = payroll.create_full_payroll(df_payroll=df_erp_payroll, df_mapping=df_map_payroll)
    
    # Full target schema
    full_columns = [
        'DocumentNumber', 'Date', 'Currency',
        'AmountDKK', 'AmountEUR', 'Type',
        'GLAccount', 'CostCenter',
        'Department', 'CustomerName',
        'ItemName', 'SourceType'
    ]

    # Reindex all ERP dataframes to align to full schema
    df_expenses_full = df_erp_expenses_full.reindex(columns=full_columns)
    df_products_full = df_erp_products_full.reindex(columns=full_columns)

    # Concatenate all ERP data
    df_erp_all = pd.concat([df_expenses_full, df_products_full], ignore_index=True)
    
    print(f"✔ All erp-data generated.")

    if save_to_csv:
        output_dir = f"data/outputdata/{company_name}/ERP"
        os.makedirs(output_dir, exist_ok=True)
        df_erp_expenses_full.to_csv(f"{output_dir}/erp_expenses.csv", index=False)
        df_erp_payroll_full.to_csv(f"{output_dir}/erp_payroll.csv", index=False)
        df_erp_products_full.to_csv(f"{output_dir}/erp_products.csv", index=False)
        df_erp_all.to_csv(f"{output_dir}/erp_all.csv", index=False)
        print(f"✔ All ERP CSVs saved to: {output_dir}")
    
    return {
        "df_erp_expenses_full": df_erp_expenses_full,
        "df_erp_payroll_full": df_erp_payroll_full,
        "df_erp_products_full": df_erp_products_full,
        "df_erp_all": df_erp_all,
    }