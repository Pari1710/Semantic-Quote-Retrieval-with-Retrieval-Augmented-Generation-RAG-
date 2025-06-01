# Semantic Quote Retrieval with Retrieval-Augmented Generation (RAG)

This project implements a semantic quote retrieval system using Retrieval-Augmented Generation (RAG) leveraging a fine-tuned sentence-transformer model and a generative language model. It enables users to query a large dataset of quotes and get relevant quotes with context-aware natural language answers.

---

## Features

- Fine-tuned sentence-transformer model for semantic search on quotes dataset
- Efficient vector similarity search using FAISS index
- Natural language answer generation using Google Flan-T5 model conditioned on retrieved quotes
- Interactive Streamlit app with:
  - Query input
  - Generated answer display
  - Top retrieved quotes with similarity scores
  - Visualizations of author and tag distributions
  - JSON export of query results

---

## Dataset

The project uses the [Abirate English Quotes dataset](https://huggingface.co/datasets/Abirate/english_quotes) from HuggingFace.

---

## Setup and Usage

### Requirements

- Python 3.8+
- Install dependencies:
```bash
pip install -r requirements.txt
Run Jupyter Notebook
Prepare and fine-tune the sentence-transformer model on the quotes dataset.

Generate embeddings and build the FAISS index.

Test the RAG pipeline with example queries.

Run manual evaluation scripts.

Run Streamlit App

streamlit run app.py
Open your browser at http://localhost:8501 and enter your query.

Project Structure
notebooks/ — Jupyter notebooks for data prep, fine-tuning, indexing, and evaluation

app.py — Streamlit application code for interactive quote retrieval and visualization

texts.pkl — Saved preprocessed quote texts

corpus_embeddings.npy — Precomputed quote embeddings for FAISS

requirements.txt — Python dependencies list

Evaluation
Manual evaluation was performed using precision@k metrics on multiple test queries, confirming strong retrieval relevance. Export scripts for evaluation results are included for further analysis.

Insights and Challenges
Fine-tuning embeddings improves semantic retrieval quality

RAG combines retrieval accuracy with generative flexibility

Visualizations help analyze author/tag distribution in results

API limits required dummy models for some evaluation metrics

Prompt size constraints require careful context selection

Future Work
Integrate automated RAG evaluation frameworks (e.g., RAGAS, Quotient)

Enhance UI with additional visual analytics (quote length, sentiment)

Optimize indexing for larger datasets

Explore advanced LLMs for answer generation

License
This project is licensed under the MIT License.


