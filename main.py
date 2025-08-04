import sys
from pathlib import Path
import os
# Add project root to sys.path
sys.path.append(str(Path().resolve().parent))  # Adjust if needed

import numpy as np
import pandas as pd

from generators.full_generators import (
    create_company_data
)

company_name = "Vestas"
count_products = 50
count_employees = 50

data = create_company_data(company_name, count_employee = 50, count_products = 50, count_accounts = 30, count_departments = 10, count_customers = 10, save_to_csv=True)