import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "dbname": "ai_job_tracker",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    return conn