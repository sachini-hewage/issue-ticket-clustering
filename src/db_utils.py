# src/db_utils.py
import os
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text, inspect
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




def fetch_and_cleanup_tickets(ticket_ids):
    """
    Fetch specified tickets from the 'tickets' table while:
      - Deleting related records from 'ticket_preprocessed' and 'ticket_embeddings'
      - Clearing cluster-related fields in the 'tickets' table.

    Parameters
    ticket_ids : list of str
        List of ticket IDs to fetch.

    Returns
    pandas.DataFrame
        DataFrame containing fetched ticket records.
    """
    if not ticket_ids:
        raise ValueError("ticket_ids list cannot be empty")

    # Ensure tuple formatting for SQL IN clause
    ticket_tuple = tuple(ticket_ids)

    with engine.begin() as conn:  # `begin()` automatically commits or rolls back

        # Delete from ticket_preprocessed
        conn.execute(text("""
            DELETE FROM ticket_preprocessed
            WHERE ticket_id IN :ticket_ids;
        """), {"ticket_ids": ticket_tuple})

        # Delete from ticket_embeddings
        conn.execute(text("""
            DELETE FROM ticket_embeddings
            WHERE ticket_id IN :ticket_ids;
        """), {"ticket_ids": ticket_tuple})

        # Clear cluster-related fields in tickets table
        conn.execute(text("""
            UPDATE tickets
            SET 
                cluster_id = NULL,
                cluster_label = NULL,
                next_step = NULL,
                cluster_confidence = NULL,
                neighbour_confidence = NULL
            WHERE ticket_id IN :ticket_ids;
        """), {"ticket_ids": ticket_tuple})
        
        # Fetch the tickets
        query_fetch = text("""
            SELECT * FROM tickets
            WHERE ticket_id IN :ticket_ids;
        """)
        df = pd.read_sql(query_fetch, conn, params={"ticket_ids": ticket_tuple})

    return df





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




def write_tickets_to_ticket_preprocessed(df,table_name="ticket_preprocessed", if_exists="append"):
    """
    Write a pandas DataFrame to a database table.

    Args:
        df (pd.DataFrame): DataFrame to write.
        db_url (str): SQLAlchemy database URL (e.g., 'postgresql://user:pass@host:port/dbname').
        table_name (str): Name of the table in the database.
        if_exists (str): What to do if table exists. Options: 'fail', 'replace', 'append'.

    Returns:
        None
    """

    # Write DataFrame to the table
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"[INFO] DataFrame written to table '{table_name}' successfully.")


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



def update_ticket_with_cluster_and_recommendation(ticket_id, cluster_id, cluster_label, suggestion, cluster_confidence, neighbour_confidence):
    """
    Update the tickets table with cluster_id, cluster_label, next_step, and cluster_confidence and neighbour_confidence.
    Automatically adds missing columns if they don't exist.
    """
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("tickets")]

    # Identify missing columns
    alter_stmts = []
    if "cluster_id" not in columns:
        alter_stmts.append("ALTER TABLE tickets ADD COLUMN cluster_id INTEGER;")
    if "cluster_label" not in columns:
        alter_stmts.append("ALTER TABLE tickets ADD COLUMN cluster_label TEXT;")
    if "next_step" not in columns:
        alter_stmts.append("ALTER TABLE tickets ADD COLUMN next_step TEXT;")
    if "cluster_confidence" not in columns:
        alter_stmts.append("ALTER TABLE tickets ADD COLUMN cluster_confidence FLOAT;")
    if "neighbour_confidence" not in columns:
        alter_stmts.append("ALTER TABLE tickets ADD COLUMN neighbour_confidence FLOAT;")

    # Add missing columns
    if alter_stmts:
        with engine.begin() as conn:
            for stmt in alter_stmts:
                conn.execute(text(stmt))
        print(f"[INFO] Added missing columns: {', '.join([stmt.split()[3] for stmt in alter_stmts])}")

    # Update record with new information
    query = text("""
        UPDATE tickets
        SET cluster_id = :cluster_id,
            cluster_label = :cluster_label,
            next_step = :suggestion,
            cluster_confidence = :cluster_confidence,
            neighbour_confidence = :neighbour_confidence
        WHERE ticket_id = :ticket_id
    """)

    with engine.begin() as conn:
        conn.execute(query, {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "suggestion": suggestion,
            "cluster_confidence": cluster_confidence,
            "neighbour_confidence" : neighbour_confidence,
            "ticket_id": ticket_id
        })

    print(f"Table \"tickets\" updated with {ticket_id}'s cluster_id, cluster_label, suggestion and confidences.")

