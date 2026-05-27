import polars as pl
from pathlib import Path
from tqdm import tqdm

from telegram_quality_control.db import (
    get_conn_string,
    recreate_database,
    get_engine,
    apply_constraints,
)

import os
from dotenv import load_dotenv

load_dotenv()

raw_data_folder = Path(os.getenv("RAW_DATA_FOLDER", "raw_data"))

db_name = os.getenv("DB_NAME")

# 1. Apply pre-data schema (create tables, don't set constraints/indexes)
recreate_database(db_name, raw_data_folder / "schema_only.sql")

db_url = get_conn_string(database=db_name)

# 2. Load parquet files table by table
for path in raw_data_folder.iterdir():
    if path.is_dir():
        table_name = path.stem
        print(f"Loading data for table '{table_name}'...")
        for parquet_file in tqdm(sorted(path.glob("*.parquet"))):
            df = pl.read_parquet(parquet_file)
            df.write_database(table_name, db_url, if_table_exists="append")
    elif path.is_file() and path.suffix == ".parquet":
        table_name = path.stem
        print(f"Loading data for table '{path.name}'...")
        df = pl.read_parquet(path)
        df.write_database(table_name, db_url, if_table_exists="append")

# 3. Apply post-data schema (constraints, indexes, FK)
constraints_file = raw_data_folder / "constraints.sql"
engine = get_engine(db_name)
apply_constraints(engine, constraints_file)
