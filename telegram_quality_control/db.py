from dotenv import dotenv_values
from sqlalchemy import create_engine, text


def get_conn_string(credentials=".env", database=None):
    config = dotenv_values(credentials)

    db_user = config["DB_USER"]
    db_pass = config["DB_PASSWORD"]
    db_host = config["DB_HOST"]
    db_port = config["DB_PORT"]
    if database is None:
        db_name = config["DB_NAME"]
    else:
        db_name = database

    db_url = f'postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'

    return db_url


def check_if_exists(db_name):
    base_url = get_conn_string(database="postgres")
    engine = create_engine(base_url, isolation_level='AUTOCOMMIT')

    with engine.connect() as conn:
        # Check if database exists
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db_name"), {"db_name": db_name}
        )
        exists = result.fetchone() is not None
    
    engine.dispose()
    
    return exists


def create_database(db_name):
    """Create the database if it doesn't exist."""
    # Connect to the default 'postgres' database to create new database
    base_url = get_conn_string(database="postgres")
    exists = check_if_exists(db_name)

    if not exists:
        engine = create_engine(base_url, isolation_level='AUTOCOMMIT')
        with engine.connect() as conn:
            # Create the database
            conn.execute(text(f'CREATE DATABASE {db_name}'))
            print(f"Database '{db_name}' created successfully!")
        engine.dispose()
    else:
        print(f"Database '{db_name}' already exists.")


def delete_database(db_name):
    """Delete the database if it exists."""
    # Connect to the default 'postgres' database to drop the target database
    
    exists = check_if_exists(db_name)
    if not exists:
        print(f"Database '{db_name}' does not exist. No action taken.")
        return

    base_url = get_conn_string(database="postgres")
    engine = create_engine(base_url, isolation_level='AUTOCOMMIT')
    
    with engine.connect() as conn:
        # Terminate existing connections to the database
        conn.execute(
            text(
                f"""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = :db_name AND pid <> pg_backend_pid()
        """
            ),
            {"db_name": db_name},
        )

        # Drop the database
        conn.execute(text(f'DROP DATABASE {db_name}'))
        print(f"Database '{db_name}' deleted successfully!")

    engine.dispose()


def create_tables(engine, schema_file):
    """Create all tables defined in the pg_dump file."""

    with open(schema_file) as f:
        query = '\n'.join(line for line in f if not line.startswith("\\"))

    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()

    # check if tables are created
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        )
        tables = [row[0] for row in result]

    print("All tables created successfully!")
    print("Tables in the database:", tables)


def recreate_database(db_name, schema_file):
    """Delete the database if it exists, then create a new one."""
    print("Recreating database...")
    exists = check_if_exists(db_name)
    if exists:
        response = input(f"Database '{db_name}' already exists. Should it be deleted and recreated? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return

    delete_database(db_name)
    create_database(db_name)

    engine = get_engine(db_name)
    create_tables(engine, schema_file)

    print("\nDatabase setup complete!")
    return engine


def apply_constraints(engine, constraints_file):
    with open(constraints_file) as f:
        query = '\n'.join(line for line in f if not line.startswith("\\"))

    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()
        
    return engine

def get_engine(db_name):
    """Create and return the engine for the new database."""
    database_url = get_conn_string(database=db_name)
    engine = create_engine(database_url, pool_size=5, max_overflow=10)
    return engine
