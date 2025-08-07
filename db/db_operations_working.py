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


def insert_dataframe_from_csv(
    csv_path: str, table_name: str, schemadict: dict, conn=None, version_tag=None, final_dim_dict=None
):
    """
    Insert data from a CSV file into a specified table in Microsoft SQL Server.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file.
    table_name : str
        The name of the table in Microsoft SQL Server.
    conn : pyodbc.Connection, optional
        An existing connection to the database. If None, a new connection will be created.

    Returns
    -------
    None
    """
    

    version_id_query = f"SELECT id FROM dbo.dim_version WHERE version = '{version_tag}'"
    version_id = conn.cursor().execute(version_id_query).fetchone()[0]

    df = load_dataframe_from_csv(csv_path, schemadict[table_name])
    df["id"] = df.index
    df = optimize_datatypes(df)
    df["version_id"] = version_id  # Add version_id column to the DataFrame
    tbl_statement = create_table_statement(df, table_name)
    table_exists_result = table_exists(table_name, conn=conn)
    if not table_exists_result:
        print(tbl_statement)
        conn.cursor().execute(tbl_statement)
        conn.commit()

    max_id_query = f"SELECT MAX(id) FROM {table_name}"
    max_id = conn.cursor().execute(max_id_query).fetchone()[0]
    if max_id is None:
        max_id = 0  # If the table is empty, start from 0
    print(f"Max ID in {table_name}: {max_id}")
    df["id"] = max_id + df.index + 1
    insert_dataframe(df, table_name, conn=conn)
    final_dim_dict[table_name] = df
    
def remap_dataframe_ids(df: pd.DataFrame, dim_dict: dict) -> pd.DataFrame:
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


def execute_db_operations(version_tag: str = "test6"):
    
    final_dim_dict = {}
    print(dotenv.get_key(".env", "SERVER"))
    print(dotenv.get_key(".env", "DATABASE"))
    print(dotenv.get_key(".env", "UID"))
    print(dotenv.get_key(".env", "PWD"))
    conn = connect_to_database()

    version_query = f"INSERT INTO dbo.dim_version (version) VALUES ('{version_tag}')"
    print(version_query)
    conn.cursor().execute(version_query)  # Ensure the connection is established
    conn.commit()

    for file_name in [
        "department.csv",
        "customer.csv",
        "product.csv",
        "account.csv",
        "procurement.csv",
        "service.csv",
        "line.csv",
        #"employee.csv",
    ]:
        print(f"inserting {file_name} into database")
        table_name = "dim_" + file_name.split(".")[0].replace(" ", "_").lower()

        insert_dataframe_from_csv(
            csv_path=f"data/outputdata/dimensions/{file_name}",
            table_name=table_name,
            schemadict=schemadict,
            conn=conn,
            version_tag=version_tag,
            final_dim_dict=final_dim_dict,
        )
    
    df_erp = pd.read_csv("data/outputdata/fact/general_ledger.csv")
    df_erp_mapped = remap_dataframe_ids(df_erp, final_dim_dict)
    df_erp_mapped.to_csv("data/outputdata/fact/general_ledger_mapped.csv", index=False)

    #mapping = dict(zip(final_dim_dict['dim_product']['product_name'], final_dim_dict['dim_product']['id']))
    #df_erp['product_id'] = df_erp['product_id'].map(mapping)
    print("Mapping complete for general ledger.")
    print("Inserting payroll data...")

    df_payroll = pd.read_csv("data/outputdata/fact/erp_payroll.csv")
    df_payroll_mapped = remap_dataframe_ids(df_payroll, final_dim_dict)
    df_payroll_mapped.to_csv("data/outputdata/fact/erp_payroll_mapped.csv", index=False)

    df_employee = pd.read_csv("data/outputdata/dimensions/employee.csv")
    df_employee = remap_dataframe_ids(df_employee, final_dim_dict)
    df_employee.to_csv("data/outputdata/dimensions/employee_mapped.csv", index=False)

    insert_dataframe_from_csv(
        csv_path="data/outputdata/dimensions/employee_mapped.csv",
        table_name="dim_employee",
        schemadict=schemadict,
        conn=conn,
        version_tag=version_tag,
        final_dim_dict=final_dim_dict,
    )

    insert_dataframe_from_csv(
        csv_path="data/outputdata/fact/erp_payroll_mapped.csv",
        table_name="fact_payroll",
        schemadict=schemadict,
        conn=conn,
        version_tag=version_tag,
        final_dim_dict=final_dim_dict,
    )

    insert_dataframe_from_csv(
        csv_path="data/outputdata/fact/general_ledger_mapped.csv",
        table_name="fact_general_ledger",
        schemadict=schemadict,
        conn=conn,
        version_tag=version_tag,
        final_dim_dict=final_dim_dict,
    )

if __name__ == "__main__":

    final_dim_dict = {}
    print(dotenv.get_key(".env", "SERVER"))
    print(dotenv.get_key(".env", "DATABASE"))
    print(dotenv.get_key(".env", "UID"))
    print(dotenv.get_key(".env", "PWD"))
    conn = connect_to_database()

    version_query = f"INSERT INTO dbo.dim_version (version) VALUES ('test6')"
    print(version_query)
    conn.cursor().execute(version_query)  # Ensure the connection is established
    conn.commit()

    for file_name in [
        "department.csv",
        "customer.csv",
        "product.csv",
        "account.csv",
        "procurement.csv",
        "service.csv",
        "line.csv",
        #"employee.csv",
    ]:
        print(f"inserting {file_name} into database")
        table_name = "dim_" + file_name.split(".")[0].replace(" ", "_").lower()

        insert_dataframe_from_csv(
            csv_path=f"data/outputdata/dimensions/{file_name}",
            table_name=table_name,
            schemadict=schemadict,
            conn=conn,
            version_tag=version_tag,
        )
    
    #print(final_dim_dict["dim_account"])
    #print(final_dim_dict)
    #df_erp = pd.read_csv("data/outputdata/fact/general_ledger.csv")
    #df_erp_mapped = remap_dataframe_ids(df_erp, final_dim_dict)
    #df_erp_mapped.to_csv("data/outputdata/fact/general_ledger_mapped.csv", index=False)

    #mapping = dict(zip(final_dim_dict['dim_product']['product_name'], final_dim_dict['dim_product']['id']))
    #df_erp['product_id'] = df_erp['product_id'].map(mapping)


    df_payroll = pd.read_csv("data/outputdata/fact/erp_payroll.csv")
    df_payroll_mapped = remap_dataframe_ids(df_payroll, final_dim_dict)
    df_payroll_mapped.to_csv("data/outputdata/fact/erp_payroll_mapped.csv", index=False)


  #  insert_dataframe_from_csv(
  #      csv_path="data/outputdata/fact/general_ledger_mapped.csv",
  #      table_name="fact_general_ledger",
  #      schemadict=schemadict,
  #      conn=conn,
  #      version_tag="test7",
 #   )

