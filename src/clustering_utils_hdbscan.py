# src/clustering_utils.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from src.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
from src.ai_utils import generate_llm_summary
import datetime

# Create reusable DB engine for SQLAlchemy
def get_engine():
    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)


# Fetch embeddings and corresponding ticket_id
def fetch_embeddings(limit=None):
    """
    Fetch ticket embeddings and their text for clustering.
    Only demo_flag = FALSE data is used.
    """
    engine = get_engine()
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = text(f"""
        SELECT te.ticket_id, te.embedding, tp.combined_text
        FROM ticket_embeddings te
        JOIN ticket_preprocessed tp ON te.ticket_id = tp.ticket_id
        WHERE tp.combined_text IS NOT NULL
        {limit_clause}
    """)
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(query).fetchall(), columns=["ticket_id", "embedding", "combined_text"])
    # Convert embedding text (e.g. "[0.1, -0.2, ...]") into Python list
    df["embedding"] = df["embedding"].apply(lambda x: list(map(float, x.strip("[]").split(","))))
    return df


# Dimensionality Reduction (UMAP)
def reduce_dimensions(embeddings, n_components=20, n_neighbors=15, metric='cosine'):
    """
    Reduce high-dimensional embeddings using UMAP.

    Parameters:
        embeddings: np.array of embeddings
        n_components: target dimensions after reduction
        n_neighbors: how many nearest neighbors to consider for local structure
        metric: distance measure (cosine works best for text embeddings)
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,      # Controls local vs global structure (smaller = finer local clusters)
        n_components=n_components,    # Target dimension (e.g., 10–50)
        metric=metric,                # Cosine = good for sentence-transformer embeddings
        #random_state=42
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced, reducer


# Cluster using HDBSCAN
def cluster_embeddings(reduced_embeddings, min_cluster_size=10, metric='euclidean'):
    """
    Cluster reduced embeddings using HDBSCAN.

    Parameters:
        min_cluster_size: minimum size for a cluster (smaller = more clusters)
        metric: distance measure for clustering
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,          # Merge small similar clusters
        min_samples=25,               # Slightly less strict on noise
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.1,  # Merge nearby clusters
        prediction_data=True
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    return labels, clusterer






def soft_assign_noise(reduced_embeddings, labels, similarity_threshold=0.7):
    """
    Reassign HDBSCAN noise points (-1) to nearest cluster if similarity >= threshold.

    Parameters:
        reduced_embeddings: np.array of reduced embeddings (after UMAP)
        labels: array of HDBSCAN labels (-1 = noise)
        similarity_threshold: min cosine similarity to assign to a cluster

    Returns:
        new_labels: array of updated cluster labels
    """
    new_labels = labels.copy()
    noise_mask = labels == -1
    cluster_mask = labels != -1

    if noise_mask.sum() == 0:
        return new_labels  # No noise points

    # Compute centroids of existing clusters
    cluster_ids = np.unique(labels[cluster_mask])
    centroids = np.vstack([reduced_embeddings[labels == cid].mean(axis=0) for cid in cluster_ids])

    # Cosine similarity between noise points and centroids
    similarities = cosine_similarity(reduced_embeddings[noise_mask], centroids)

    # Find best cluster per noise point
    best_idx = similarities.argmax(axis=1)
    best_sim = similarities.max(axis=1)

    # Assign to cluster if similarity >= threshold
    for i, sim in enumerate(best_sim):
        if sim >= similarity_threshold:
            new_labels[np.where(noise_mask)[0][i]] = cluster_ids[best_idx[i]]
        # else remains -1

    return new_labels



# Store results in DB
def save_clusters(df_clusters):
    """
    Save cluster labels to ticket_embeddings table.
    Adds/updates a 'cluster_id' column.
    """
    engine = get_engine()
    with engine.begin() as conn:
        # Add cluster_id column if not exists
        conn.execute(text("""
            ALTER TABLE ticket_embeddings 
            ADD COLUMN IF NOT EXISTS cluster_id INTEGER
        """))
        # Update rows
        for _, row in df_clusters.iterrows():
            conn.execute(
                text("""
                    UPDATE ticket_embeddings
                    SET cluster_id = :cluster_id
                    WHERE ticket_id = :ticket_id
                """),
                {"cluster_id": int(row["cluster_id"]), "ticket_id": row["ticket_id"]}
            )



# Find Top-N representative tickets of a cluster

def get_cluster_representatives(df, reduced_embeddings, top_n=10):
    """
    Return top N tickets closest to cluster centroid for each cluster.
    """
    representatives = {}
    df = df.reset_index(drop=True)
    embeddings = reduced_embeddings

    for cluster_id in df["cluster_id"].unique():
        if cluster_id == -1:
            continue  # skip noise

        mask = df["cluster_id"] == cluster_id
        cluster_embeddings = embeddings[mask.values]

        centroid = cluster_embeddings.mean(axis=0)
        sims = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
        top_idx = sims.argsort()[-top_n:][::-1]
        top_ticket_ids = df.loc[mask].iloc[top_idx]["ticket_id"].tolist()

        representatives[cluster_id] = top_ticket_ids

    return representatives



def label_clusters_llm(df, representatives, max_tickets=10):
    """
    Generate natural language labels for each cluster using an LLM.

    Parameters:
        df: DataFrame with ['ticket_id', 'combined_text', 'cluster_id']
        representatives: dict {cluster_id: [ticket_ids]} mapping cluster to representative tickets
        max_tickets: maximum number of tickets to include per cluster for prompt

    Returns:
        cluster_labels: dict {cluster_id: label_text}
    """
    cluster_labels = {}

    for cluster_id, ticket_ids in representatives.items():
        # Take only the top N tickets
        ticket_ids = ticket_ids[:max_tickets]
        texts = df[df["ticket_id"].isin(ticket_ids)]["combined_text"].tolist()

        # Join the ticket texts into a single string
        texts_combined = "\n".join(texts)

        # Build the prompt for LLM
        prompt = f"""Here are {len(texts)} example ticket texts from cluster {cluster_id}:
{texts_combined}

Provide a concise, descriptive label summarizing the common theme of these tickets using 3-6 words.
Return only the label.
"""

        try:
            label = generate_llm_summary(prompt)
            cluster_labels[cluster_id] = label
            print(f"Cluster {cluster_id}: {label}")
        except Exception as e:
            print(f"Error generating label for cluster {cluster_id}: {e}")
            cluster_labels[cluster_id] = None

    return cluster_labels





# Save cluster labels to DB

def save_cluster_labels(cluster_labels):
    """Save cluster labels into ticket_embeddings table."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE ticket_embeddings 
            ADD COLUMN IF NOT EXISTS cluster_label TEXT
        """))
        for cluster_id, label in cluster_labels.items():
            conn.execute(
                text("""
                    UPDATE ticket_embeddings
                    SET cluster_label = :label
                    WHERE cluster_id = :cluster_id
                """),
                {"label": label, "cluster_id": int(cluster_id)}
            )




def save_cluster_labels_to_table(cluster_labels: dict):
    """
    Save or update cluster labels generated by the LLM into Postgres.

    Parameters:
        cluster_labels (dict): {cluster_id: label_text}
    """
    engine = get_engine()
    with engine.begin() as conn:
        # Ensure table exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cluster_labels (
                cluster_id INTEGER PRIMARY KEY,
                label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Upsert (insert or update) cluster labels
        for cluster_id, label in cluster_labels.items():
            conn.execute(
                text("""
                    INSERT INTO cluster_labels (cluster_id, label, created_at)
                    VALUES (:cid, :lbl, :ts)
                    ON CONFLICT (cluster_id)
                    DO UPDATE SET label = EXCLUDED.label, created_at = EXCLUDED.created_at
                """),
                {"cid": int(cluster_id), "lbl": label, "ts": datetime.datetime.utcnow()}
            )

