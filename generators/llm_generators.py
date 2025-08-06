import pandas as pd 
import utils.prompt_utils as prompt_utils
import utils.utils as utils
import generators.random_generators as random_generators

def generate_procurement_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.8):
    client = prompt_utils.get_openai_client()

    over_request_count = int(count) * 1.2
    header = "name;Proportionality"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)

    prompt = f"""
    You are a procurement and industry expert. Your task is to generate a realistic ranked list of the top {over_request_count} procurement items, materials, and consumables 
    commonly purchased by a company like {company_name}, based on its industry and typical operations.

    Each row should contain:
    - name: A specific, realistic name of the purchased item or material
    - Proportionality: an estimation of how much of the total procurement budget is spent on this item, as a percentage (0-100).

    The list must cover the full range of procurement, including:
    - Raw materials and base components used in the company’s production
    - Operational and maintenance supplies
    - General-purpose equipment and consumables
    - Standardized office and administrative products

    Ensure that:
    - The beginning of the list contains the most expensive raw materials or specialized production inputs
    - The middle of the list reflects typical tools, spare parts, and operational items
    - The last items consist of low-cost, generic office and administrative supplies

    Rank the entire list by Proportionality in descending order.

    {constraints}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data analyst and industry expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )
    df_procurement = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_procurement = utils.convert_column_to_percentage(df_procurement, "Proportionality", scale=1.0)
    return df_procurement

def generate_sales_products_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 1):
    """
    Generates a list of products that the company sells, with proportionality scores.
    The products are categorical and realistic to the company's industry and operations.

    Args:
        company_name (str): Name of the company (used for industry context).
        count (int): Number of products to generate.
        model (str): OpenAI model to use.
        temp (float): Temperature setting for generation.

    Returns:
        pd.DataFrame: A DataFrame with columns 'ProductName' and 'Proportionality'.
    """
    client = prompt_utils.get_openai_client()

    over_request_count = int(count * 1.2)
    header = "name;Proportionality"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)

    prompt = f"""
    You are a product marketing and industry expert. Your task is to generate a realistic ranked list of the top {over_request_count} products
    that a company like {company_name} would sell, based on its industry, brand identity, and market focus.

    Each row should contain:
    - name: A specific, realistic name of a sellable product or SKU category
    - Proportionality: An estimation of how much of the total sales revenue is attributed to this product, as a percentage (0-100)

    Ensure that:
    - Products reflect typical B2B or B2C outputs for a company in the relevant industry
    - There is a natural mix of high-revenue flagship items, mid-range products, and low-cost accessories or services
    - Product names should remain categorical, not overly specific (e.g., “Premium Hiking Boots” or “Small Business CRM Plan”)

    Rank the entire list by Proportionality in descending order.

    {constraints}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data analyst and industry expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )

    df_products = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_products = utils.convert_column_to_percentage(df_products, "Proportionality", scale=1.0)
    return df_products


def generate_services_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.5):
    client = prompt_utils.get_openai_client()

    over_request_count = int(count) * 1.2
    header = "name;Proportionality"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)

    prompt = f"""
    You are a finance and procurement expert. Your task is to generate a realistic ranked list of the top {over_request_count} services, licenses, and fees 
    commonly incurred by a company like {company_name}, based on its industry and typical operations.

    Each row should contain:
    - name: A realistic and specific name of a fee or service (e.g. 'IT Consulting', 'Cleaning Contract', 'Microsoft 365 License', 'Legal Retainer')
    - Proportionality: an estimation of how much of the total procurement budget is spent on this item, as a percentage (0-100).

    The list should include a broad mix of services such as:
    - One-time fees (e.g. onboarding, legal review, security audits)
    - Recurring contracts (e.g. accounting, IT maintenance, janitorial services)
    - Software licenses and subscriptions (e.g. Microsoft 365, Power BI Pro, Zoom, Adobe Acrobat)
    - General professional services (e.g. recruitment agency fees, compliance advisory)
    Do not make anual entries

    Ensure that:
    - The **top** of the list contains expensive project-based services and enterprise retainers
    - The **middle** contains regular professional services and departmental support
    - The **bottom entries** are standardized, lower-cost software licenses and administrative services (e.g. antivirus subscriptions, file storage plans, domain hosting, video conferencing tools)

    Rank the list by descending Proportionality.

    {constraints}
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data analyst and finance expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )
    df_services = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_services = utils.convert_column_to_percentage(df_services, "Proportionality", scale=1.0)
    return df_services

def generate_roles_llm(company_name: str, count: int = 100, model: str = "gpt-4o", temp: float = 0.1):
    client = prompt_utils.get_openai_client()

    over_request_count = int(count) * 1.2
    header = "RoleName"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)

    prompt = f"""
    You are a human resources and industry expert. Your task is to generate a realistic list of {over_request_count} employees in a company like {company_name}, 
    based on its industry and typical operations.

    Each row should contain:
    - RoleName: The job title or position of the employee

    The output should reflect realistic HR data for financial modeling and analytics purposes. 
    It should be ranked by the highest monthly salary.

    {constraints}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data analyst and HR expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )

    return prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)

def generate_names_llm(count: int = 100, model: str = "gpt-4o", temp: float = 0.1):
    client = prompt_utils.get_openai_client()

    # Over-request
    over_request_count = int(count) * 1.2

    header = "FirstName;LastName"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)

    prompt = f"""
    You are a data assistant. Your task is to generate {over_request_count} realistic and common Danish full names. 
    The names should be common Danish names, suitable for a diverse population. 

    Set the split between the sexes to 50/50, and ensure that the names are diverse and representative of a Danish population. 
    They should sound like they are between 30-60 years old — no trendy names.

    {constraints}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )

    return  prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)


def generate_accounts_llm(company_name: str, count: int = 30, model: str = "gpt-4o", temp: float=0.1) -> pd.DataFrame:
    """
    Generate a realistic Chart of Accounts (COA) for a company using LLMs.
    """

    client = prompt_utils.get_openai_client()
    over_request_count = int(count) * 1.2

    header = "name;AccountType"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)

    prompt = f"""
    You are a financial accountant and ERP systems expert. Your task is to generate a realistic Chart of Accounts (COA)
    for a company like {company_name}, suitable for use in general ledger data, ERP demos, and financial modeling.

    For each row, return:
    - name: a descriptive name of the account (e.g., "Sales Revenue", "Consulting Fees", "Bank Account", etc.)
    - AccountType: the type of the account — must be one of the following categories:
    "Revenue", "Product Expense", "Service Expense", "Payroll", "Asset", or "Equity"

    Ensure the following:
    - Include a diverse mix of accounts, including both P&L and balance sheet categories.
    - Clearly separate expense accounts into:
    * "Product Expense" — raw materials, packaging, freight-in, etc.
    * "Service Expense" — consulting, software subscriptions, legal fees, etc.
    * "Payroll" — salaries, bonuses, pension contributions, etc.

    Include a diverse mix of accounts (e.g., income, overhead, balance sheet accounts). Avoid duplicates.
    {constraints}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful ERP and accounting assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )

    df_accounts = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_accounts["Account_ID"] = random_generators.generate_account_ids(df_accounts["AccountType"])

    return df_accounts

def generate_departments_llm(company_name: str, count: int = 10, model: str = "gpt-4o", temp: float = 0.3):
    client = prompt_utils.get_openai_client()

    header = "name;Proportionality"
    constraints = prompt_utils.get_standard_constraints(header, count)

    prompt = f"""
    You are an HR and workforce distribution expert. Your task is to generate {count} realistic departments for a company like {company_name}, 
    including their location and relative share of total payroll.

    For each department, provide the following fields in CSV format:
    - name: A realistic department name (e.g. R&D, Sales, Finance, Customer Support)
    - Proportionality: An estimated proportion of the company's total payroll allocated to this department, expressed as a decimal (e.g. 0.25 for 25%)

    The total Proportionality values should sum to approximately 1.0 across all departments.

    Departments should vary in size and function, including both strategic and operational units.

    {constraints}
    """
    ### Recomendation: 
    # Model = "gpt-4o"  Medium reasoning
    # temperature = 0.3 Low creativity
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful real estate and HR data assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )

    df_offices = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_offices.insert(0, "Department_ID", range(100, len(df_offices) + 100))
    df_offices = utils.convert_column_to_percentage(df_offices, "Proportionality", scale=1.0)
    return df_offices


def generate_customers_llm(company_name: str, count: int = 100, model: str = "gpt-4o", temp: float = 0.3):
    client = prompt_utils.get_openai_client()

    header = "name;CustomerSegment;Proportionality"
    constraints = prompt_utils.get_standard_constraints(header, count)

    prompt = f"""
    You are a B2B sales and marketing expert. Your task is to generate a list of {count} realistic customers for a company like {company_name}.
    These customers should reflect typical business clients based on the company's industry, geography, and operations.

    For each customer, provide the following fields in CSV format:
    - name: The name of the customer (company or organization)
    - CustomerSegment: One of the following segments: Enterprise, SME, Government, Non-profit, Retail, Wholesale, or Startup
    - Proportionality: An estimated proportion of the company's total revenue generated by this customer, expressed as a decimal (e.g. 0.05 for 5%)

    The Proportionality values should vary realistically across customers and sum to approximately 1.0 in total.
    Ensure variation in customer size and segment, including both large key accounts and smaller clients.

    {constraints}
    """
    
    ### Recomendation: 
    # Model = "gpt-4o"  Medium reasoning
    # temperature = 0.3 Low creativity

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data assistant and B2B customer segmentation expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )

    df_customers = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_customers.insert(0, "Customer_ID", range(10, len(df_customers) + 10))
    df_customers = utils.convert_column_to_percentage(df_customers, "Proportionality", scale=1.0)
    return df_customers


def estimate_mean_pay_llm(company_name: str, model: str = "gpt-4o", temp: float = 0):
    client = prompt_utils.get_openai_client()

    prompt = f"""
    What is the mean pay for a {company_name} employee in DKK? Include pension, vacation, and benefits. It should be a realistic estimate based on the company's industry and typical operations.
    Output just a number — the mean monthly salary in DKK. No text or units, do not divide the number by a comma or period.
    Output should be a single integer, rounded to the nearest whole number.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful payroll analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )

    return int(response.choices[0].message.content)

def estimate_financials_llm(company_name: str, model: str = "gpt-4o", temp: float = 0):
    client = prompt_utils.get_openai_client()

    prompt = f"""
    You are a financial controller.

    Estimate the **annual total revenue** and **operating costs** for a company like "{company_name}" in DKK. Only the danish finances should be estimated.
    Base it on the company's typical industry, geography, and scale.
    Output format (no explanation):
    <total_finances>

    Output just a number — the added annual revenue and operating costs. No text or units, do not divide the number by a comma or period.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful payroll analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )

    total_finances = int(response.choices[0].message.content)

    # We assume the following split for each 
    # Allocate 35% to payroll, 45% to products, 15% to services, 5% to overhead. 
    payroll = int(total_finances * 0.35)
    products = int(total_finances * 0.45)
    services = int(total_finances * 0.15)
    overhead = int(total_finances * 0.05)

    final_finances = {
        "total_finances": total_finances,
        "payroll": payroll,
        "products": products,
        "services": services,
        "overhead": overhead
    }
    return final_finances