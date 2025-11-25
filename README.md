# Multilingual Ticket Classification & Next-Step Recommendation

This is an end-to-end pipeline for classifying multilingual support tickets, assigning them to semantic clusters, and generating recommended next actions based on historical patterns. 
The project combines preprocessing, multilingual embeddings, cluster lookup, and diagnostic tooling into a single reproducible workflow.

---

## Overview

This repository contains the inference pipeline responsible for:

1. **Intake of raw multilingual tickets**
2. **Text preprocessing and normalization**
3. **Embedding generation** using sentence embeddings
4. **Cluster assignment** via pre-computed vector clusters
5. **Next-step recommendation** derived from historical resolutions
6. **Diagnostic visualizations** for model validation and explainability

The entire process is implemented in:

**inference_pipeline.ipynb**

This project is intended as a personal technical exploration and is designed for extensibility and experimentation.

---

## Key Features

### Multilingual Support
- Automatic language detection
- Optional translation to a common pivot language (English)
- Consistent embedding space for EN, SV, FI, and others

### Semantic Embedding Layer
- Supports SentenceTransformers or local embedding models
- Produces dense vector representations optimized for clustering

### Cluster-Based Classification
- Assigns new tickets to existing semantic clusters
- Retrieves the nearest past tickets for interpretability
- Detects outliers and previously unseen ticket types

### Next-Step Recommendation
- Leverages past resolutions and knowledge-base actions
- Returns structured recommended actions for new tickets
- Includes confidence and nearest-neighbor evidence

### Diagnostics & Visualization
- Compare new tickets to historical examples
- Visualize clusters using dimensionality reduction (UMAP)
- Highlight cluster neighbors for interpretability


---

## Workflow Summary

### 1. Load Tickets
Unseen tickets are selected (e.g., via `demo_flag = TRUE` in PostgreSQL).

### 2. Preprocessing
- Normalization
- Language detection
- Optional translation
- Cleaning/token-level preparation

### 3. Embedding Generation
Converts text into dense vector form using a multilingual embedding model.

### 4. Cluster Assignment
- Finds the nearest cluster centroid
- Retrieves semantically similar historical tickets

### 5. Recommendation Stage
Generates next-step actions informed by previous resolutions associated with the cluster.

### 6. Diagnostics
Includes tooling for:
- Neighbor retrieval
- Cluster visualization
- Embedding projection for sanity checks

### 7. Database Write-Back
Final classification and recommendations are stored linked to the original ticket ID.

---

## Requirements
All requirements can be found in requirements.txt file on which you can utilise pip install -r requirements.txt for installation. 


