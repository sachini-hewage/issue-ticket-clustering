"""
src/clustering_utils_keywords.py

This module clusters tickets based purely on their extracted keywords
(from the ticket_preprocessed table). It connects to the Postgres DB via SQLAlchemy,
vectorizes keyword lists using TF-IDF, clusters them with HDBSCAN or KMeans,
and writes cluster IDs back into the ticket_embeddings table.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import hdbscan
from sqlalchemy import text
from sqlalchemy import create_engine, text


from src.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

# Create reusable DB engine for SQLAlchemy
def get_engine():
    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)

# Fetch keywords
def fetch_keywords():
    """
    Fetch ticket_id and keywords from the ticket_preprocessed table.

    - keywords may be stored as a string like "['login', 'password']"
      or as a space-separated string ("login password").
    - This normalizes them into a single string for TF-IDF.
    """
    engine = get_engine()
    query = text("""
        SELECT ticket_id, keywords
        FROM ticket_preprocessed
        WHERE keywords IS NOT NULL
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    # Normalize keywords to text
    df["keywords_str"] = df["keywords"].apply(
        lambda kw: " ".join(eval(kw)) if isinstance(kw, str) and kw.startswith("[") else str(kw)
    )

    return df



# TF-IDF Vectorization

def vectorize_keywords(df):
    """
    Vectorize keywords into TF-IDF feature space.

    Parameters:
        df : DataFrame containing 'keywords_str'

    Returns:
        X : sparse matrix (n_tickets x n_features)
        vectorizer : trained TfidfVectorizer object
    """
    vectorizer = TfidfVectorizer(
        analyzer='word',        # Treat each word as a term
        token_pattern=r'\w+',   # Include alphanumeric tokens
        lowercase=True,         # Normalize case
        max_features=200       # Cap vocabulary size for efficiency
    )

    X = vectorizer.fit_transform(df["keywords_str"])
    return X, vectorizer



def cluster_keywords(X, method="hdbscan", n_clusters=10, min_cluster_size=5):
    """
    Cluster TF-IDF keyword vectors.

    Parameters:
        X : TF-IDF matrix
        method : 'kmeans' or 'hdbscan'
        n_clusters : number of clusters (for KMeans)
        min_cluster_size : minimum size for HDBSCAN clusters

    Returns:
        labels : cluster assignments for each sample
        model : fitted clustering model
    """
    if method == "kmeans":
        # KMeans - fixed number of clusters
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto"
        )
        labels = model.fit_predict(X)

    else:
        # HDBSCAN - hierarchical density-based clustering
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,   # Minimum cluster size
            metric='euclidean',                  # Use Euclidean for full compatibility
            cluster_selection_method='eom',
            prediction_data=True
        )
        labels = model.fit_predict(X.toarray())  # Convert sparse → dense matrix

    return labels, model



# Save cluster IDs
def save_keyword_clusters(df):
    """
    Save keyword-based cluster labels to the ticket_embeddings table.

    - Adds a column 'keyword_cluster_id' if it does not exist.
    - Updates each ticket_id row with its new cluster assignment.
    """
    engine = get_engine()
    with engine.begin() as conn:
        # Ensure column exists
        conn.execute(text("""
            ALTER TABLE ticket_embeddings 
            ADD COLUMN IF NOT EXISTS keyword_cluster_id INTEGER
        """))

        # Update each row's cluster ID
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    UPDATE ticket_embeddings
                    SET keyword_cluster_id = :cluster_id
                    WHERE ticket_id = :ticket_id
                """),
                {
                    "cluster_id": int(row["cluster_id"]) if row["cluster_id"] != -1 else None,
                    "ticket_id": row["ticket_id"]
                }
            )

    print("Keyword-based cluster IDs saved to 'ticket_embeddings'.")



# Orchestrator

def run_keyword_clustering(method="hdbscan", n_clusters=200, min_cluster_size=5):
    """
    Full pipeline:
      1. Fetch keywords
      2. Vectorize
      3. Cluster
      4. Save results to DB
    """
    print("Fetching keywords from DB...")
    df = fetch_keywords()

    print(f"Retrieved {len(df)} tickets with keywords.")

    print("Vectorizing keywords using TF-IDF...")
    X, _ = vectorize_keywords(df)

    print(f"Clustering using {method.upper()}...")
    labels, model = cluster_keywords(X, method, n_clusters, min_cluster_size)

    df["cluster_id"] = labels
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {num_clusters} clusters (and {sum(labels == -1)} noise points).")

    print("Saving results to database...")
    save_keyword_clusters(df)

    print("Keyword clustering completed successfully.")
    return df, model
