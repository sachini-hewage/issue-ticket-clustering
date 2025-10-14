# src/embeddings.py
from sentence_transformers import SentenceTransformer
from src.config import LOCAL_EMBEDDING_MODEL, BATCH_SIZE
from typing import List

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
