# src/preprocessing.py
import re
import torch
import spacy
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import MarianTokenizer, MarianMTModel
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# MODEL SETUP

# English spaCy model for Named Entity Recognition (NER)
EN_NLP = spacy.load("en_core_web_sm")


# Translation models (Helsinki-NLP)
TRANSLATION_MODELS = {
    "sv": "Helsinki-NLP/opus-mt-sv-en",
    "fi": "Helsinki-NLP/opus-mt-fi-en"
}

# Cache so the translation model is only loaded once per session
TRANSLATION_CACHE = {}

# Load keyword detection model 
KEYBERT_MODEL = None


# LANGUAGE DETECTION
def detect_language(text):
    """
    Detects the language of a given text using langdetect.
    Returns a short ISO language code (e.g., 'en', 'sv', 'fi').
    If text is empty or ambiguous, returns 'unknown'.
    """
    if not text or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# TEXT CLEANING
def clean_text(text):
    """
    Cleans input text by:
      - Removing emojis and pictographs
      - Collapsing multiple spaces into one
      - Stripping leading/trailing spaces
    """
    if not text:
        return ""

    # Regex pattern to remove emojis and other pictorial Unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map icons
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002B00-\U00002BFF"  # arrows
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    # Collapse multiple spaces → single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    return text.strip()


# TRANSLATION (batch-ready, MPS-safe)
def translate_texts_batch(texts, lang, batch_size=32):
    """
    Translate a list of texts to English using Helsinki-NLP models.
    Returns a list of translated texts in the same order.
    Only translates if a model exists for the language.

    Splits into smaller batches for MPS/CPU to avoid out-of-memory errors.
    Logs after each batch is processed.
    """
    if lang not in TRANSLATION_MODELS:
        return texts  # either English or unknown, skip translation

    model_name = TRANSLATION_MODELS[lang]

    # Load and cache model/tokenizer
    if model_name not in TRANSLATION_CACHE:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        if torch.backends.mps.is_available():
            model = model.to("mps")
        elif torch.cuda.is_available():
            model = model.to("cuda")
        
        TRANSLATION_CACHE[model_name] = (tokenizer, model)
    else:
        tokenizer, model = TRANSLATION_CACHE[model_name]
    
    device = next(model.parameters()).device
    translated_texts = []

    # Helper to split into smaller batches
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for batch_idx, text_batch in enumerate(chunks(texts, batch_size), start=1):
        inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        translated_tokens = model.generate(**inputs)
        batch_translated = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        translated_texts.extend(batch_translated)

        # Log batch completion
        print(f"[INFO] Translated batch {batch_idx} ({len(text_batch)} texts) for language '{lang}'")

        # Free MPS memory cache if needed
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return translated_texts


# PII MASKING
def mask_pii_en(text):
    """
    Masks personally identifiable information (PII) in English text.

    This is two-fold:
      1. English spaCy NER (for PERSON, GPE)
      2. Regex-based masking for:
          - Email addresses
          - IBANs
          - Credit card numbers
          - Phone numbers
    """
    if not text:
        return ""

    masked_text = text

    # NER-based masking  for names and locations
    doc = EN_NLP(masked_text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE"]:
            placeholder = f"[{ent.label_}]"
            masked_text = masked_text.replace(ent.text, placeholder)

    # Regex-based masking

    # Emails: name@domain.tld
    masked_text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "[EMAIL]",
        masked_text
    )

    # IBANs: 2 letters + 2 digits + 10–30 alphanumeric characters (spaces optional)
    masked_text = re.sub(
        r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}\b",
        "[IBAN]",
        masked_text
    )

    # Credit cards: 13–16 digits, optional spaces/dashes between
    masked_text = re.sub(
        r"\b(?:\d[ -]?){13,16}\b",
        "[CARD]",
        masked_text
    )

    # Phone numbers: +XX XXX XXX XXXX or compact +XXXXXXXXXXXX
    masked_text = re.sub(
        r"\+?\d[\d\s\-]{7,}\d",
        "[PHONE]",
        masked_text
    )

    return masked_text



# KEYWORD EXTRACTION (using KeyBERT)
from tqdm import tqdm

def extract_keywords_batch(texts, top_n=10, log_every=1000):
    """
    Extracts meaningful keywords from a list of texts using KeyBERT, 
    skipping known placeholders like [PERSON], [GPE], [IBAN], etc.
    Returns a list of keyword lists.

    Added:
      - tqdm progress bar for large datasets
      - periodic print logs every `log_every` items
    """
    global KEYBERT_MODEL

    if KEYBERT_MODEL is None:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        KEYBERT_MODEL = KeyBERT(model=embedder)

    all_keywords = []

    # tqdm progress bar for visual tracking
    for i, text in enumerate(tqdm(texts, desc="Extracting keywords", unit="text")):
        if not text or len(text.strip()) == 0:
            all_keywords.append([])
            continue

        # Remove placeholders so they don't appear as keywords
        clean_text_ = re.sub(r'\[(PERSON|GPE|IBAN|EMAIL|CARD|PHONE)\]', '', text)
        clean_text_ = re.sub(r'\s+', ' ', clean_text_).strip()

        # Extract keywords
        keywords = KEYBERT_MODEL.extract_keywords(
            clean_text_,
            keyphrase_ngram_range=(1, 4),
            stop_words='english',
            top_n=top_n
        )
        all_keywords.append([kw[0] for kw in keywords])

        # Periodic logging for large datasets
        if (i + 1) % log_every == 0:
            print(f"[INFO] Processed {i + 1}/{len(texts)} texts")

    print(f"[INFO] Keyword extraction completed for {len(texts)} texts")
    return all_keywords

