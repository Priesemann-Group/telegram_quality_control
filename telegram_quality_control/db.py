from dotenv import dotenv_values

def get_conn_string(credentials=".env"):
    config = dotenv_values(credentials)
    
    db_user = config["DB_USER"]
    db_pass = config["DB_PASSWORD"]
    db_host = config["DB_HOST"]
    db_port = config["DB_PORT"]
    db_name = config["DB_NAME"]
    
    db_url = f'postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
    
    return db_url

