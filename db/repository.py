from db.config import get_db_connection

def insert_job(title: str, company: str, description: str, embedding: list):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO jobs (title, company, description, embedding) VALUES (%s, %s, %s, %s) RETURNING id;",
        (title, company, description, embedding)
    )
    job_id = cur.fetchone()['id']
    conn.commit()
    cur.close()
    conn.close()
    return job_id

def get_similar_jobs(embedding: list, limit: int = 3):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, company, description, (1 - (embedding <=> %s::vector)) as similarity FROM jobs ORDER BY embedding <=> %s::vector LIMIT %s;",
        (embedding, embedding, limit)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows