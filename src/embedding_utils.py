# src/embeddings.py
from sentence_transformers import SentenceTransformer
from src.config import LOCAL_EMBEDDING_MODEL, BATCH_SIZE
from typing import List
from src.db_utils import fetch_preprocessed_tickets, fetch_already_embedded_ticket_ids, insert_embeddings


# Load model once
model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)



def embed_texts_local(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    return embeddings.tolist()

def batch_embed_and_store(tickets: List[dict], insert_fn):
    """
    Embed tickets and store embeddings in DB using insert_fn.
    tickets: list of dicts with 'ticket_id' and 'combined_text'
    insert_fn: function to save embeddings
    """
    n = len(tickets)
    print(f"Embedding {n} tickets (batch size {BATCH_SIZE})")
    for i in range(0, n, BATCH_SIZE):
        batch = tickets[i:i+BATCH_SIZE]
        ids = [t["ticket_id"] for t in batch]
        texts = [t["combined_text"] if t["combined_text"].strip() else " " for t in batch]
        embs = embed_texts_local(texts)
        records = [(tid, list(map(float, emb))) for tid, emb in zip(ids, embs)]
        insert_fn(records)
        print(f"Inserted embeddings for batch {i}-{i+len(batch)-1}")


from src.db_utils import fetch_preprocessed_tickets, fetch_already_embedded_ticket_ids, insert_embeddings
from src.embedding_utils import batch_embed_and_store



def embed_new_tickets(limit=None, dry_run=False):
    """
    Fetch preprocessed tickets, filter out already embedded ones, and embed remaining tickets.
    
    Args:
        limit (int, optional): Maximum number of tickets to fetch. Defaults to None (all).
        dry_run (bool, optional): If True, do not insert embeddings; just print info. Defaults to False.
    
    Returns:
        int: Number of tickets embedded in this run.
    """
    # Fetch tickets
    tickets = fetch_preprocessed_tickets(limit=limit)
    if not tickets:
        print("[INFO] No tickets fetched.")
        return 0

    # Filter out already embedded tickets
    already = fetch_already_embedded_ticket_ids()
    to_embed = [t for t in tickets if t["ticket_id"] not in already]
    print(f"[INFO] {len(to_embed)} tickets left to embed")

    if not to_embed:
        return 0

    # Define insertion function
    if dry_run:
        def insert_fn(records):
            print(f"[DRY RUN] Would insert {len(records)} embeddings; sample id: {records[0][0]}")
    else:
        insert_fn = insert_embeddings

    # Run batch embedding and store
    batch_embed_and_store(to_embed, insert_fn)

    print(f"Embeddings for {len(to_embed)} tickets added to \"ticket_embeddings\" table.")

