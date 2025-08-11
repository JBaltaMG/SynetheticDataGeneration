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
count_product = 50
count_employee = 50

data = create_company_data(company_name, count_employee = 50, count_product = 50, count_account = 30, count_department = 10, count_customer = 10, save_to_csv=True)


# add costumer_id, vendor_id, 
# rename pay to department_id,

