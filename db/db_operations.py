import pyodbc
import dotenv
import pandas as pd
import numpy as np
from db.schema import schemadict

# Load environment variables from .env file
dotenv.load_dotenv()


def connect_to_database():
    """
    Establish a connection to the SQL Server database.
    Returns a connection object.
    """
    servername = dotenv.get_key(".env", "SERVER")
    database = dotenv.get_key(".env", "DATABASE")
    username = dotenv.get_key(".env", "UID")
    password = dotenv.get_key(".env", "PWD")
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER=tcp:{servername},1433;"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate=Yes;"
        f"Encrypt=yes;"  # Optional, but recommended for security
    )
    print(f"Connection string: {connection_string}")
    try:
        connection = pyodbc.connect(connection_string)
        return connection
    except pyodbc.Error as e:
        print(f"Error connecting to database: {e}")
        return None


def load_dataframe_from_csv(csv_path: str, schema: dict) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    df = pd.read_csv(csv_path, dtype=schema)
    df.columns = [col.lower() for col in df.columns]
    # df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def map_dtype(dtype):
    """
    Map Pandas data type to corresponding Microsoft SQL Server data type.
    """
    if str(dtype) in ("int8", "int16", "int32"):
        return "int"
    elif str(dtype) == "int64":
        return "bigint"
    elif str(dtype) == "float16" or str(dtype) == "float32" or str(dtype) == "float64":
        return "decimal(18,2)"
    elif str(dtype) == "bool":
        return "bit"
    elif str(dtype).startswith("datetime64") or "datetime" in str(dtype) or str(dtype) == "datetime64[ns]":
        return "datetime"
    elif str(dtype).startswith("timedelta"):
        return "bigint"
    elif str(dtype) == "category":
        return "nvarchar(100)"
    else:
        return "nvarchar(255)"


def create_table_statement(df: pd.DataFrame, tablename: str):
    """
    Generate a "Create table" statement with the column names of the DataFrame and the corresponding Microsoft SQL Server data types.
    """
    # Get the data types of each column
    datatypes = df.dtypes

    # Map each data type to corresponding Microsoft SQL Server data type
    sql_datatypes = datatypes.apply(map_dtype)

    # Combine the column names and data types into a single list of tuples
    columns = list(zip(df.columns, sql_datatypes))

    # Build the "Create table" statement
    create_table = f"CREATE TABLE [{tablename}] ("
    for col, dtype in columns:
        create_table += f"[{col}] {dtype}, "
    create_table = create_table[:-2] + ");"

    return create_table


def table_exists(table_name, conn=None, conn_str=None):
    """
    Check if a table exists in a Microsoft SQL Server database.

    Parameters:
        conn_str (str): A string representing the connection string for the database.
        table_name (str): A string representing the name of the table you want to check for.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    # Establish a connection to the database.
    if conn is None:
        conn = pyodbc.connect(conn_str)

    # Create a cursor.
    cursor = conn.cursor()

    # Define the query to count the number of tables with the specified name.
    query = f"SELECT COUNT(*) FROM sys.tables WHERE name = '{table_name}'"

    # Execute the query.
    cursor.execute(query)

    # Fetch the result of the query.
    result = cursor.fetchone()[0]

    # Close the connection.
    # conn.close()

    # Return True if the result is greater than zero, False otherwise.
    return result > 0


def insert_dataframe(df, table_name, connection_string=None, conn=None) -> None:
    """
    Insert Pandas DataFrame into the specified table in Microsoft SQL Server.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be inserted into the table.
    table_name : str
        The name of the table in Microsoft SQL Server.
    connection_string : str
        The connection string to connect to the Microsoft SQL Server database.

    Returns
    -------
    None
    """
    # Connect to the database
    if conn is None:
        conn = pyodbc.connect(connection_string)

    if table_exists(table_name=table_name, conn=conn) is False:
        print(
            f"Table doesn't exist. Please create a table with name {table_name} before inserting data."
        )
        return
    cursor = conn.cursor()

    # Create a list of tuples representing the rows of the DataFrame
    rows = [tuple(row) for row in df.to_numpy()]

    # Insert the rows into the table
    sql = f"INSERT INTO [{table_name}] ({','.join([f'[{col}]' for col in df.columns])}) VALUES ({','.join(['?' for _ in df.columns])})"
    print(sql)
    cursor.executemany(sql, rows)

    # Commit the changes
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    # conn.close()


def optimize_datatypes(df):
    """
    Optimize the data types in the DataFrame to use the smallest possible data type for each column.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose data types are to be optimized.

    Returns
    -------
    None
    """
    for col in df.columns:
        col_type = df[col].dtype

        if np.issubdtype(col_type, np.integer):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif np.issubdtype(col_type, np.floating):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df

def insert_dataframe_from_df(df: pd.DataFrame, table_name: str, schemadict: dict, conn, version_tag=None):
    """
    Inserts a DataFrame directly into a SQL Server table, converting unsupported dtypes.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to insert.
    table_name : str
        The name of the destination SQL table.
    schemadict : dict
        Dictionary defining expected column types.
    conn : pyodbc.Connection
        Open database connection.
    version_tag : str, optional
        If provided, looks up version_id from dim_version and adds it to the DataFrame.
    """
    df.columns = df.columns.str.strip().str.lower()

    if version_tag:
        version_id_query = f"SELECT id FROM dbo.dim_version WHERE version = '{version_tag}'"
        version_id = conn.cursor().execute(version_id_query).fetchone()[0]
        df["version_id"] = version_id

    # Convert pandas nullable Int64 columns to native int64
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]) and df[col].dtype.name == "Int64":
            df[col] = df[col].fillna(-1).astype("int64")

    df = optimize_datatypes(df)

    if not table_exists(table_name, conn):
        sql_create = create_table_statement(df, table_name)
        print(f"[INFO] Creating table {table_name}:\n{sql_create}")
        conn.cursor().execute(sql_create)
        conn.commit()

    max_id = conn.cursor().execute(f"SELECT MAX(id) FROM {table_name}").fetchone()[0] or 0
    print(f"[INFO] Max ID in {table_name}: {max_id}")
    df["id"] = max_id + df.index + 1

    insert_dataframe(df, table_name, conn=conn)


def insert_dataframe_from_csv(csv_path, table_name, schemadict, conn, version_tag=None, final_dim_dict=None):
    df = load_dataframe_from_csv(csv_path, schemadict[table_name])
    df.columns = df.columns.str.strip().str.lower()

    if final_dim_dict is not None:
        final_dim_dict[table_name] = df.copy()

    if version_tag:
        version_id_query = f"SELECT id FROM dbo.dim_version WHERE version = '{version_tag}'"
        version_id = conn.cursor().execute(version_id_query).fetchone()[0]
        df["version_id"] = version_id

    df = optimize_datatypes(df)

    if not table_exists(table_name, conn):
        sql_create = create_table_statement(df, table_name)
        print(sql_create)
        conn.cursor().execute(sql_create)
        conn.commit()

    max_id = conn.cursor().execute(f"SELECT MAX(id) FROM {table_name}").fetchone()[0] or 0
    print(f"Max ID in {table_name}: {max_id}")
    df["id"] = max_id + df.index + 1

    insert_dataframe(df, table_name, conn=conn)

def remap_dataframe_ids(df: pd.DataFrame, dim_dict: dict) -> pd.DataFrame:
    """
    Efficiently remaps categorical *_id columns using provided dimension mappings.
    Assumes each dim_dict entry has 'name' and 'id' columns.
    Ignores case and whitespace when matching.
    """
    for key, item in dim_dict.items():
        if not {'name', 'id'}.issubset(item.columns):
            continue

        id_col_name = key.split("_")[-1] + "_id"
        if id_col_name in df.columns:
            mapping = dict(zip(item["name"].astype(str).str.strip().str.lower(), item["id"]))
            before = df[id_col_name].astype(str).str.strip().str.lower()

            df[id_col_name] = (
                before
                .replace(mapping)
                .infer_objects(copy=False)
                .fillna(-1)
                .astype("Int64")
            )

            # Debug: print any unmapped values
            unmapped = before[~before.isin(mapping.keys())].unique()
            if len(unmapped) > 0:
                print(f"[WARN] Unmapped values in {id_col_name} for {key}: {unmapped}")

    return df

def process_dimension_tables(files, schemadict, conn, version_tag, final_dim_dict):
    for file_name in files:
        table_name = "dim_" + file_name.split(".")[0].replace(" ", "_").lower()
        print(f"Inserting {file_name} into {table_name}...")
        insert_dataframe_from_csv(
            csv_path=f"data/outputdata/dimensions/{file_name}",
            table_name=table_name,
            schemadict=schemadict,
            conn=conn,
            version_tag=version_tag,
            final_dim_dict=final_dim_dict
        )

def process_fact_table(input_path, table_name, final_dim_dict, schemadict, conn, version_tag):
    df = pd.read_csv(input_path)
    df_mapped = remap_dataframe_ids(df, final_dim_dict)

    for col in df_mapped.columns:
        if col.endswith("_id"):
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors="coerce").fillna(-1).astype("Int64")

    insert_dataframe_from_df(
        df=df_mapped,
        table_name=table_name,
        schemadict=schemadict,
        conn=conn,
        version_tag=version_tag
    )


def remap_dataframe_ids_emp(df: pd.DataFrame, dim_dict: dict) -> pd.DataFrame:
    """
    Efficiently remaps categorical columns to ID columns using mapping dictionaries.
    """
    id_maps = {
        key: dict(zip(item['name'], item['id']))
        for key, item in dim_dict.items()
    }

    for key, mapping in id_maps.items():
        id_col_name = key.split("_")[-1] + "_id"
        if id_col_name in df.columns:
            df[id_col_name] = (
                df[id_col_name]
                .replace(mapping)
                .infer_objects(copy=False)       
                .fillna(-1)
                .astype("Int64")
            )
    return df

def handle_employee_id(input_string: str, dim_dict: dict) -> pd.DataFrame:
    """
    Handles the remapping of employee IDs in the DataFrame based on the provided dimension dictionary.
    """

    #df_employee = pd.read_csv(input_string)
    #df_employee = remap_dataframe_ids_emp(df_employee, dim_dict)
    #df_employee.to_csv("data/outputdata/dimensions/employee_mapped.csv", index=False)

    #insert_dataframe_from_csv(
    #    csv_path="data/outputdata/dimensions/employee_mapped.csv",
    #    table_name="dim_employee",
    #    schemadict=schemadict,
    #    conn=conn,
    #    version_tag=version_tag,
    #    final_dim_dict=final_dim_dict,
    #) 

def execute_db_operations(version_tag: str):
    conn = connect_to_database()

    print("SQL connection established.")
    print("SERVER:", dotenv.get_key(".env", "SERVER"))
    print("DATABASE:", dotenv.get_key(".env", "DATABASE"))

    version_insert = f"INSERT INTO dbo.dim_version (version) VALUES ('{version_tag}')"
    conn.cursor().execute(version_insert)
    conn.commit()

    final_dim_dict = {}

    dim_files = [
        "department.csv", "customer.csv", "product.csv", "account.csv",
        "procurement.csv", "service.csv", "line.csv"
    ]
    process_dimension_tables(dim_files, schemadict, conn, version_tag, final_dim_dict)

    handle_employee_id("data/outputdata/dimensions/employee.csv", final_dim_dict)

    process_fact_table(
        input_path="data/outputdata/fact/general_ledger.csv",
        table_name="fact_general_ledger",
        final_dim_dict=final_dim_dict,
        schemadict=schemadict,
        conn=conn,
        version_tag=version_tag
    )

    process_fact_table(
        input_path="data/outputdata/fact/erp_payroll.csv",
        table_name="fact_payroll",
        final_dim_dict=final_dim_dict,
        schemadict=schemadict,
        conn=conn,
        version_tag=version_tag
    )

if __name__ == "__main__":
    from datetime import datetime as dt
    company_name = "Lego"
    version_tag = company_name.lower() + dt.now().strftime("%Y%m%d%H%M%S")
    execute_db_operations(version_tag)

 