# src/db_utils.py
import os
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.config import DB_URL, BATCH_SIZE

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def get_db_credentials():
    """Return DB credentials as a dict for SQLAlchemy"""
    return {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "db_name": os.getenv("DB_NAME")
    }




engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)



def fetch_preprocessed_tickets(limit=None):
    """
    Fetch all tickets from ticket_preprocessed table.
    Assumes demo_flag is always FALSE.
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = text(f"""
        SELECT ticket_id, combined_text
        FROM ticket_preprocessed
        WHERE combined_text IS NOT NULL
        {limit_clause}
    """)
    with engine.connect() as conn:
        # Use .mappings() to get dictionary-like rows
        result = conn.execute(query)
        return [dict(row) for row in result.mappings().all()]


def fetch_already_embedded_ticket_ids():
    """Return set of ticket_ids already embedded in ticket_embeddings."""
    query = text("SELECT ticket_id FROM ticket_embeddings")
    with engine.connect() as conn:
        result = conn.execute(query)
        return {row['ticket_id'] for row in result.mappings().all()}


# def insert_embeddings(records):
#     """
#     Insert embeddings into ticket_embeddings table.
#     records: list of tuples (ticket_id, embedding_list)
#     """
#     insert_sql = text("""
#         INSERT INTO ticket_embeddings (ticket_id, embedding)
#         VALUES (:ticket_id, :embedding::vector)
#         ON CONFLICT (ticket_id) DO UPDATE
#         SET embedding = EXCLUDED.embedding,
#             created_at = NOW()
#     """)
#     with engine.begin() as conn:  # transactional batch insert
#         for i in range(0, len(records), BATCH_SIZE):
#             batch = records[i:i+BATCH_SIZE]
#             data = [{"ticket_id": tid,
#                      "embedding": "[" + ",".join(map(str, emb)) + "]"}
#                     for tid, emb in batch]
#             conn.execute(insert_sql, data)

def insert_embeddings(records):
    """
    Insert embeddings into ticket_embeddings table.
    records: list of tuples (ticket_id, embedding_list)
    """
    insert_sql = text("""
        INSERT INTO ticket_embeddings (ticket_id, embedding)
        VALUES (:ticket_id, :embedding)
        ON CONFLICT (ticket_id) DO UPDATE
        SET embedding = EXCLUDED.embedding,
            created_at = NOW()
    """)
    
    with engine.begin() as conn:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i+BATCH_SIZE]
            data = []
            for tid, emb in batch:
                emb_str = "[" + ",".join(map(str, emb)) + "]"  # Postgres expects string literal for VECTOR
                data.append({"ticket_id": tid, "embedding": emb_str})
            conn.execute(insert_sql, data)

