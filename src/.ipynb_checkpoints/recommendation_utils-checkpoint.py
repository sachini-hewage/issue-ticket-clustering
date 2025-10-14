"""
src/recommendation_utils.py

Recommends the next step for a new support ticket based on clustering,
historical resolution patterns, and internal comments similarity.
"""

import numpy as np
import pandas as pd
import pickle
import re
import os
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity
from src.config import DB_URL
from src.ai_utils import generate_llm_summary
from src.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
import datetime
import ast

# Create reusable DB engine for SQLAlchemy
def get_engine():
    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)


# Database connection
engine = create_engine(DB_URL)



# Text cleaning utility clean internal_comments array
def clean_internal_comments(comments_list):
    """
    Clean internal comments by removing tags, timestamps, and reference numbers.
    
    Example:
        ["[Ops] 2025-06-21 13:13 - Checked logs. Ref: 12345"]
        -> "Checked logs."
    """
    cleaned_comments = []
    for comment in comments_list:
        
        # Remove agent tags like [Ops], [TechOps]
        comment = re.sub(r'\[[^\]]*\]', '', comment)
        
        # Remove timestamps like 2025-06-21 13:13
        comment = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', '', comment)
        
        # Remove "Ref: 12345"
        comment = re.sub(r'Ref:\s*\d+', '', comment)
        
        # Normalize whitespace
        comment = re.sub(r'\s+', ' ', comment)
        
        cleaned_comments.append(comment.strip())
    
    return " ".join(cleaned_comments)



# Load my pretrained reducer and HDBSCAN models

# # Get absolute path to the models directory
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_DIR = os.path.join(BASE_DIR, "models")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # project root
MODEL_DIR = os.path.join(BASE_DIR, "models")

try:
    with open(os.path.join(MODEL_DIR, "hdbscan_model.pkl"), "rb") as f:
        clusterer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "umap_reducer.pkl"), "rb") as f:
        reducer = pickle.load(f)
except FileNotFoundError:
    clusterer = None
    reducer = None
    print(f"Warning: Model files not found in {MODEL_DIR}")



# Fetch ticket data

def get_ticket_embedding(ticket_id):
    """
    Fetch embedding, text, internal comments, and status for a given ticket.
    """
    query = text("""
        SELECT te.embedding, te.ticket_id, tp.combined_text, 
               t.internal_comments, t.status, te.cluster_id
        FROM ticket_embeddings te
        JOIN ticket_preprocessed tp ON te.ticket_id = tp.ticket_id
        JOIN tickets t ON te.ticket_id = t.ticket_id
        WHERE te.ticket_id = :ticket_id
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"ticket_id": ticket_id}).fetchone()

    if not row:
        raise ValueError(f"Ticket {ticket_id} not found in database.")

    
    embedding_list = ast.literal_eval(row.embedding) if isinstance(row.embedding, str) else row.embedding
    embedding = np.array(embedding_list, dtype=np.float32)

    comments_cleaned = clean_internal_comments(row.internal_comments) if row.internal_comments else ""
    return {
        "ticket_id": row.ticket_id,
        "embedding": embedding,
        "combined_text": row.combined_text,
        "comments_cleaned": comments_cleaned,
        "status": row.status,
        "cluster_id": row.cluster_id
    }



def find_recommendation(ticket_id, top_k=5):
    """
    Recommend the next step for a given ticket.

    Steps:
        1. Fetch current ticket's embedding & internal comments.
        2. Identify its cluster (predict if missing).
        3. Find the closest ticket in the same cluster.
        4. Prefer resolved/closed tickets; else use nearest open one.
        5. Compare internal comments & ask LLM for the next recommended step.
        6. Fetch cluster label from DB if available.

    Parameters:
        ticket_id (str): Ticket identifier.
        top_k (int): Number of similar tickets to consider (default=5).

    Returns:
        tuple: (suggestion_text, confidence_score, cluster_id, cluster_label, best_comments, nearest_ticket_id)
    """

    # Fetch current ticket embedding and details
    ticket = get_ticket_embedding(ticket_id)

    if ticket is None:
        return "Ticket not found.", 0.0, None, None, None, None

    # Predict cluster if missing or unassigned (-1)
    if ticket.get("cluster_id") is None or ticket["cluster_id"] == -1:
        if clusterer is None or reducer is None:
            raise RuntimeError("Clusterer or UMAP reducer not loaded. Cannot predict cluster.")
        reduced = reducer.transform([ticket["embedding"]])
        cluster_pred, _ = clusterer.approximate_predict(clusterer, reduced)
        cluster_id = int(cluster_pred[0])
        ticket["cluster_id"] = cluster_id

        # Update in DB
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE ticket_embeddings SET cluster_id = :cid WHERE ticket_id = :tid"),
                {"cid": cluster_id, "tid": ticket_id}
            )
    else:
        cluster_id = ticket["cluster_id"]

    # Retrieve neighbors from same cluster
    query = text("""
        SELECT te.ticket_id, te.embedding, t.status, t.internal_comments
        FROM ticket_embeddings te
        JOIN tickets t ON te.ticket_id = t.ticket_id
        WHERE te.cluster_id = :cid AND te.ticket_id != :tid
    """)
    with engine.connect() as conn:
        neighbors = pd.DataFrame(conn.execute(query, {"cid": cluster_id, "tid": ticket_id}))

    if neighbors.empty:
        return "No similar tickets found for recommendation.", 0.0, cluster_id, None, None, None

    # Compute cosine similarity between current ticket and neighbors
    neighbor_embs = np.vstack([
        np.array(ast.literal_eval(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32)
        for x in neighbors["embedding"]
    ])
    sims = cosine_similarity([ticket["embedding"]], neighbor_embs).flatten()
    neighbors["similarity"] = sims

    # Prefer resolved/closed neighbors, else fallback to most similar open
    resolved = neighbors[neighbors["status"].isin(["closed", "resolved"])]
    if not resolved.empty:
        best = resolved.iloc[resolved["similarity"].argmax()]
    else:
        best = neighbors.iloc[neighbors["similarity"].argmax()]

    confidence = float(best["similarity"])
    best_comments = clean_internal_comments(best["internal_comments"]) if best["internal_comments"] else ""
    nearest_ticket_id = best["ticket_id"]

    # Construct prompt for the LLM
    if best["status"] in ("closed", "resolved"):
        prompt = f"""
        You are a support assistant.
        
        The current ticket's internal comments so far are:
        {ticket['comments_cleaned']}

        The most similar RESOLVED ticket's comments are:
        {best_comments}

        Based on this, suggest the next logical action for the current ticket in one clear sentence.
        Return only the suggested action.
        """
    else:
        prompt = f"""
        You are a support assistant.
        
        The current ticket's internal comments so far are:
        {ticket['comments_cleaned']}

        The most similar OPEN ticket's comments are:
        {best_comments}

        Suggest a helpful next step aligned with the ongoing situation.
        Return only the suggested next step.
        """

    # Generate recommendation
    suggestion = generate_llm_summary(prompt)

    # Fetch cluster label from SQL table
    cluster_label = None
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT label FROM cluster_labels WHERE cluster_id = :cid"),
            {"cid": cluster_id}
        ).fetchone()
        if result:
            cluster_label = result[0]

    # Logging for transparency
    print(f"Ticket {ticket_id} → Cluster {cluster_id} ({cluster_label}) → "
          f"Nearest ticket: {nearest_ticket_id} → Recommended action: {suggestion} (confidence: {confidence:.2f})")

    return suggestion, confidence, cluster_id, cluster_label, best_comments, nearest_ticket_id

