# src/config.py
import os

from dotenv import load_dotenv

load_dotenv()  # Load .env file

# DB connection parameters from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "enfuce_ai_support")


# Construct SQLAlchemy connection string dynamically
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Embedding config
EMBED_PROVIDER = "local"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128
EMBED_DIM = 384
