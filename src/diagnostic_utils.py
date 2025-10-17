import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sqlalchemy import text, create_engine
from src.config import DB_URL
from sqlalchemy.orm import sessionmaker
import warnings

warnings.filterwarnings("ignore", module="umap")


engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)



def fetch_ticket_pair_details(ticket_id, neighbour_id, cluster_id):
    """
    Fetch the subject_translated and body_translated for a given ticket and its neighbour.

    Parameters:
        ticket_id (str): Main ticket ID.
        neighbour_id (str): Neighbour ticket ID.
        cluster_id (int): Cluster to which the tickets belong.
        engine: SQLAlchemy engine connected to your DB.

    Returns:
        pd.DataFrame: DataFrame containing ticket_id, subject_translated, and body_translated for both tickets.
    """
    query = text("""
        SELECT 
            p.ticket_id,
            p.subject_translated,
            p.body_translated
        FROM 
            ticket_preprocessed p
        JOIN 
            ticket_embeddings e 
        ON 
            p.ticket_id = e.ticket_id
        WHERE 
            e.cluster_id = :cluster_id
            AND p.keywords IS NOT NULL
            AND p.ticket_id IN (:ticket_id, :neighbour_id)
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "cluster_id": cluster_id,
            "ticket_id": ticket_id,
            "neighbour_id": neighbour_id
        })
    
    return df



def plot_all_clusters(ticket_id, n_neighbors=30, min_dist=0.0, metric="cosine"):
    """
    Plot a 2D UMAP projection of all clusters, highlighting a given ticket and its neighbour.

    Parameters:
        ticket_id (str): Main ticket to highlight.
        n_neighbors (int): UMAP n_neighbors parameter.
        min_dist (float): UMAP min_dist parameter.
        metric (str): Distance metric for UMAP.
    """
    # Load all embeddings with cluster info
    query = text("""
        SELECT ticket_id, embedding, cluster_id
        FROM ticket_embeddings
        WHERE cluster_id != -1
    """)
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(query).fetchall(),
                          columns=["ticket_id", "embedding", "cluster_id"])

    if df.empty:
        print("No ticket embeddings found.")
        return

    # Convert embeddings from string to numpy array if needed
    df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))

    # UMAP 2D reduction
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    reduced = reducer.fit_transform(np.vstack(df["embedding"]))
    df["x"], df["y"] = reduced[:, 0], reduced[:, 1]

    plt.figure(figsize=(12, 8))

    # Plot each cluster in a different color
    clusters = df["cluster_id"].unique()
    colors = plt.cm.get_cmap("tab20", len(clusters))

    for i, cluster in enumerate(clusters):
        cluster_points = df[df["cluster_id"] == cluster]
        base_size = 30
        point_size = max(base_size, int(3000 / len(cluster_points)))
        plt.scatter(cluster_points["x"], cluster_points["y"], s=point_size, alpha=0.6, 
                    color=colors(i), label=f"Cluster {cluster}")

    # Highlight main ticket
    target_row = df[df["ticket_id"] == ticket_id]
    if not target_row.empty:
        plt.scatter(target_row["x"], target_row["y"], c="red", s=100, marker="*", edgecolor="black", label="Target Ticket")

    plt.title("UMAP 2D Projection of All Clusters")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.tight_layout()
    plt.show()






def plot_cluster_projection(ticket_id, neighbour_id, target_cluster, n_neighbors=30, min_dist=0.0, metric="cosine"):
    """
    Plot a 2D UMAP projection of a specified cluster, highlighting a given ticket and its neighbour.

    Parameters:
        ticket_id (str): Main ticket to highlight.
        neighbour_id (str): Neighbour ticket to highlight.
        target_cluster (int): Cluster ID to plot.
        n_neighbors (int): UMAP n_neighbors parameter.
        min_dist (float): UMAP min_dist parameter.
        metric (str): Distance metric for UMAP.
    """
    # Load embeddings for tickets in the cluster
    query = text("""
        SELECT ticket_id, embedding, cluster_id
        FROM ticket_embeddings
        WHERE cluster_id = :cid
    """)
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(query, {"cid": target_cluster}).fetchall(),
                          columns=["ticket_id", "embedding", "cluster_id"])

    if df.empty:
        print(f"No tickets found for cluster {target_cluster}.")
        return

    # Convert embeddings from string to numpy array if needed
    df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))

    # UMAP 2D reduction
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    reduced = reducer.fit_transform(np.vstack(df["embedding"]))
    df["x"], df["y"] = reduced[:, 0], reduced[:, 1]

    # Adjust point size based on cluster size (keep clusters visible)
    base_size = 40
    point_size = max(base_size, int(5000 / len(df)))  # smaller clusters get larger points

    plt.figure(figsize=(10, 7))
    plt.scatter(df["x"], df["y"], s=point_size, alpha=0.7, c="skyblue", label=f"Cluster {target_cluster}")

    # Highlight main ticket
    target_row = df[df["ticket_id"] == ticket_id]
    if not target_row.empty:
        plt.scatter(target_row["x"], target_row["y"], c="red", s=point_size*2.5,
                    marker="*", edgecolor="black", label="Target Ticket")

    # Highlight neighbour ticket
    neighbour_row = df[df["ticket_id"] == neighbour_id]
    if not neighbour_row.empty:
        plt.scatter(neighbour_row["x"], neighbour_row["y"], c="green", s=point_size*2,
                    marker="P", edgecolor="black", label="Neighbour Ticket")

    plt.legend()
    plt.title(f"UMAP 2D Projection — Cluster {target_cluster}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.show()
