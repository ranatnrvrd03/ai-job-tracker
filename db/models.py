import psycopg2
from db.config import DB_CONFIG, get_db_connection

def init_db():
    sys_config = DB_CONFIG.copy()
    sys_config["dbname"] = "postgres"
    
    sys_conn = psycopg2.connect(**sys_config)
    sys_conn.autocommit = True
    sys_cur = sys_conn.cursor()
    
    sys_cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'ai_job_tracker';")
    exists = sys_cur.fetchone()
    
    if not exists:
        sys_cur.execute("CREATE DATABASE ai_job_tracker;")
        
    sys_cur.close()
    sys_conn.close()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id SERIAL PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            company VARCHAR(255),
            description TEXT NOT NULL,
            skills TEXT[],
            embedding vector(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()