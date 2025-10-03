import dotenv
import pandas as pd
import numpy as np
from datetime import datetime as dt
from sqlalchemy import create_engine
import urllib
from schemas.schema_sql import schemadict
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

def get_max_id_safe(conn, table_name: str, schema: str = "dbo") -> int:
    try:
        # Qualify with schema to be explicit
        res = conn.execute(text(f"SELECT MAX(id) FROM {schema}.{table_name}"))
        row = res.fetchone()
        return row[0] if row and row[0] is not None else 0
    except ProgrammingError as e:
        # Table missing → treat as empty
        if "Invalid object name" in str(e):
            return 0
        raise


def get_sqlalchemy_engine():
    servername = dotenv.get_key(".env", "SERVER")
    database = dotenv.get_key(".env", "DATABASE")
    username = dotenv.get_key(".env", "UID")
    password = dotenv.get_key(".env", "PWD")

    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={servername};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate=Yes;"
        f"Encrypt=yes;"
    )

    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)

def optimize_datatypes(df):
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df

def remap_dataframe_ids(df: pd.DataFrame, dim_dict: dict) -> pd.DataFrame:
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
                .apply(pd.to_numeric, errors='coerce')
                .fillna(-1)
                .astype("Int64")
            )
    return df

def insert_dataframe(df: pd.DataFrame, table_name: str, engine) -> None:
    print(f"[INFO] Inserting {len(df)} rows into {table_name} using to_sql...")
    df.to_sql(
        name=table_name,
        con=engine,
        schema="dbo",
        index=False,
        if_exists="append",
        chunksize=2_000,
        dtype=schemadict.get(table_name) 
    )


def insert_dataframe_from_csv(
    csv_path: str,
    table_name: str,
    schemadict: dict,
    engine,
    version_tag: str = None,
    final_dim_dict: dict = None,
    schema: str = "dbo",
) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]
    df = optimize_datatypes(df)

    with engine.begin() as conn:
        # get version_id safely (fail fast if missing)
        row = conn.execute(
            text("SELECT id FROM dbo.dim_version WHERE version = :v"),
            {"v": version_tag}
        ).fetchone()
        if not row or row[0] is None:
            raise ValueError(f"version_tag '{version_tag}' not found in dbo.dim_version")
        version_id = int(row[0])

        # compute new ids (ok if table doesn't exist yet)
        max_id = get_max_id_safe(conn, table_name, schema=schema)
        df.insert(0, "id", range(max_id + 1, max_id + 1 + len(df)))
        df["version_id"] = version_id
        # ensure integer dtypes
        df["id"] = df["id"].astype("int64")
        df["version_id"] = df["version_id"].astype("int64")

    # to_sql will auto-create the table if it doesn't exist (schema inferred)
    insert_dataframe(df, table_name, engine)

    if final_dim_dict is not None:
        # store a copy to avoid accidental external mutation
        final_dim_dict[table_name] = df.copy()

def insert_dataframe_from_csv_fact(csv_path: str, table_name: str, schemadict: dict, engine, version_tag=None):
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]
    df["id"] = df.index  # will be overwritten
    df = optimize_datatypes(df)

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM dbo.dim_version WHERE version = :v"),
            {"v": version_tag}
        ).fetchone()
        if not row:
            raise ValueError(f"version_tag '{version_tag}' not found in dbo.dim_version")

        df["version_id"] = int(row[0])

        max_id = get_max_id_safe(conn, table_name, schema="dbo")
        df["id"] = (max_id + df.index + 1).astype("int64")
        df["version_id"] = df["version_id"].astype("int64")

    insert_dataframe(df, table_name, engine)

def execute_db_operations(version_tag: str = "test_version"):
    final_dim_dict = {}
    engine = get_sqlalchemy_engine()

    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO dbo.dim_version (version) VALUES ('{version_tag}')"))

    dimension_files = [
        "department.csv", "customer.csv", "product.csv", "account.csv",
        "procurement.csv", "service.csv", "line.csv", "vendor.csv", "bu.csv"
    ]

    for file_name in dimension_files:
        table_name = "dim_" + file_name.split(".")[0].replace(" ", "_").lower()
        insert_dataframe_from_csv(
            csv_path=f"data/outputdata/dimensions/{file_name}",
            table_name=table_name,
            schemadict=schemadict,
            engine=engine,
            version_tag=version_tag,
            final_dim_dict=final_dim_dict
        )

    df_employee = pd.read_csv("data/outputdata/dimensions/employee.csv")
    df_employee = remap_dataframe_ids(df_employee, final_dim_dict)
    df_employee.to_csv("data/outputdata/dimensions/employee_mapped.csv", index=False)

    insert_dataframe_from_csv_fact(
        csv_path="data/outputdata/dimensions/employee_mapped.csv",
        table_name="dim_employee",
        schemadict=schemadict,
        engine=engine,
        version_tag=version_tag,
    )

    df_payroll = pd.read_csv("data/outputdata/fact/erp_payroll.csv")
    df_payroll = remap_dataframe_ids(df_payroll, final_dim_dict)
    df_payroll.to_csv("data/outputdata/fact/erp_payroll_mapped.csv", index=False)
    
    insert_dataframe_from_csv_fact(
        csv_path="data/outputdata/fact/erp_payroll_mapped.csv",
        table_name="fact_payroll",
        schemadict=schemadict,
        engine=engine,
        version_tag=version_tag
    )

    df_erp = pd.read_csv("data/outputdata/fact/general_ledger.csv")
    df_erp = remap_dataframe_ids(df_erp, final_dim_dict)
    df_erp.to_csv("data/outputdata/fact/general_ledger_mapped.csv", index=False)

    insert_dataframe_from_csv_fact(
        csv_path="data/outputdata/fact/general_ledger_mapped.csv",
        table_name="fact_general_ledger",
        schemadict=schemadict,
        engine=engine,
        version_tag=version_tag
    )

    insert_dataframe_from_csv_fact(
        csv_path="data/outputdata/fact/fact_reporting.csv",
        table_name="fact_reporting",
        schemadict=schemadict,
        engine=engine,
        version_tag=version_tag
    )

if __name__ == "__main__":
    version_tag = "demo_" + dt.now().strftime("%Y%m%d_%H%M")
    execute_db_operations(version_tag=version_tag)
