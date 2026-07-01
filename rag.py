from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np

try:
    import faiss
except ModuleNotFoundError:
    faiss = None
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


@dataclass
class LocalRAGIndex:
    chunks: List[Dict[str, str]]
    vectorizer: TfidfVectorizer
    faiss_index: object | None
    dense_vectors: np.ndarray


def clean_text(text: str) -> str:
    text = str(text).replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
    """Split TXT content into small overlapping chunks."""
    text = clean_text(text)
    chunks = []

    if not text:
        return chunks

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        start = end - overlap

    return chunks


def load_txt_files(data_dir: Path) -> List[Dict[str, str]]:
    chunks = []
    for txt_file in sorted(data_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        chunks.extend(chunk_text(text, txt_file.name))
    return chunks


def load_excel_files(data_dir: Path) -> List[Dict[str, str]]:
    """
    Read Excel files locally with pandas/openpyxl.
    Each row becomes one searchable chunk.
    """
    chunks = []

    for xlsx_file in sorted(data_dir.glob("*.xlsx")):
        sheets = pd.read_excel(xlsx_file, sheet_name=None)

        for sheet_name, df in sheets.items():
            df = df.fillna("")

            for row_number, row in df.iterrows():
                parts = [f"Sheet: {sheet_name}"]
                for col_name, value in row.items():
                    value = clean_text(value)
                    if value:
                        parts.append(f"{col_name}: {value}")

                row_text = " | ".join(parts)
                if row_text.strip():
                    chunks.append(
                        {
                            "text": row_text,
                            "source": f"{xlsx_file.name}, row {row_number + 2}"
                        }
                    )

    return chunks


def load_company_chunks(data_dir: Path = DATA_DIR) -> List[Dict[str, str]]:
    chunks = []
    chunks.extend(load_txt_files(data_dir))
    chunks.extend(load_excel_files(data_dir))

    if not chunks:
        raise ValueError("No .txt or .xlsx files found inside the data/ folder.")

    return chunks


def build_local_rag_index(data_dir: Path = DATA_DIR) -> LocalRAGIndex:
    """
    Startup pipeline:
    1. Load local TXT and Excel data
    2. Chunk the data
    3. Create local TF-IDF embeddings
    4. Store vectors in FAISS
    """
    chunks = load_company_chunks(data_dir)
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=4096
    )

    tfidf_matrix = vectorizer.fit_transform(texts).astype("float32")
    dense_vectors = tfidf_matrix.toarray()

    # Normalize vectors so inner product works like cosine similarity.
    norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True) + 1e-12
    dense_vectors = dense_vectors / norms

    faiss_index = None
    if faiss is not None:
        faiss_index = faiss.IndexFlatIP(dense_vectors.shape[1])
        faiss_index.add(dense_vectors.astype("float32"))

    return LocalRAGIndex(
        chunks=chunks,
        vectorizer=vectorizer,
        faiss_index=faiss_index,
        dense_vectors=dense_vectors.astype("float32")
    )


def retrieve_top_k(rag_index: LocalRAGIndex, question: str, k: int = 3) -> List[Dict[str, str | float]]:
    question_vector = rag_index.vectorizer.transform([question]).astype("float32").toarray()

    if question_vector.shape[1] == 0:
        return []

    question_norm = np.linalg.norm(question_vector, axis=1, keepdims=True) + 1e-12
    question_vector = question_vector / question_norm

    if rag_index.faiss_index is not None:
        scores, indices = rag_index.faiss_index.search(question_vector.astype("float32"), k)
        scores = scores[0]
        indices = indices[0]
    else:
        all_scores = (rag_index.dense_vectors @ question_vector.T).reshape(-1)
        indices = all_scores.argsort()[::-1][:k]
        scores = all_scores[indices]

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        chunk = rag_index.chunks[int(idx)]
        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "score": float(score)
            }
        )

    return results


def split_answer_units(text: str) -> List[str]:
    """
    Split chunks into answer units.
    Excel rows remain as complete units. TXT chunks are split into sentences.
    """
    if " | " in text:
        return [text]

    units = re.split(r"(?<=[.!?])\s+", text)
    return [unit.strip() for unit in units if len(unit.strip()) > 20]



def parse_excel_row(row_text: str) -> Dict[str, str]:
    """Convert an Excel row chunk into a dictionary."""
    data = {}
    for part in row_text.split(" | "):
        if ":" in part:
            key, value = part.split(":", 1)
            data[key.strip()] = value.strip()
    return data


def format_product_answer(row_text: str, question: str) -> str:
    """Create a cleaner answer from one product Excel row."""
    data = parse_excel_row(row_text)
    q = question.lower()

    name = data.get("Product Name", "This product")
    category = data.get("Category", "")
    price = data.get("Monthly Price USD", "")
    support = data.get("Warranty / Support", "")
    availability = data.get("Availability", "")
    description = data.get("Description", "")

    if "price" in q or "cost" in q or "monthly" in q:
        if price:
            return f"{name} monthly price is {price} USD."

    if "cybersecurity" in q or "monitoring" in q:
        return f"{name} is the product used for {category.lower()} monitoring. {description}"

    if "availability" in q or "available" in q:
        if availability:
            return f"{name} availability is {availability}."

    if "support" in q or "warranty" in q:
        if support:
            return f"{name} support/warranty information: {support}."

    details = []
    if category:
        details.append(f"Category: {category}")
    if price:
        details.append(f"Monthly Price USD: {price}")
    if availability:
        details.append(f"Availability: {availability}")
    if description:
        details.append(f"Description: {description}")

    return f"{name} — " + " | ".join(details)

def create_extractive_answer(question: str, retrieved_chunks: List[Dict[str, str | float]]) -> str:
    """
    Local no-API answer generator.
    It selects the most relevant sentences/rows from retrieved chunks.
    If relevance is weak, it says I don't know.
    """
    if not retrieved_chunks:
        return "I don't know"

    top_score = float(retrieved_chunks[0]["score"])

    # With TF-IDF, zero/very-low score usually means the topic is not in the data.
    if top_score < 0.08:
        return "I don't know"

    combined_context = " ".join(str(chunk["text"]).lower() for chunk in retrieved_chunks)
    q_lower = question.lower()

    # If the user asks for a specific field that does not exist in the retrieved data,
    # return the required fallback instead of guessing from general company text.
    specific_missing_terms = [
        "ceo", "founder", "owner", "director", "headquarters",
        "phone", "contact number", "revenue", "profit", "salary"
    ]
    for term in specific_missing_terms:
        if term in q_lower and term not in combined_context:
            return "I don't know"

    product_intent_terms = [
        "product", "price", "cost", "monthly", "category", "availability",
        "available", "warranty", "support", "cybersecurity", "monitoring",
        "autoflow", "cloudmove", "securewatch", "insightdash", "clientcare"
    ]
    excel_chunks = [chunk for chunk in retrieved_chunks if "Sheet:" in str(chunk["text"])]
    if excel_chunks and any(term in q_lower for term in product_intent_terms):
        return format_product_answer(str(excel_chunks[0]["text"]), question)

    answer_units = []
    for chunk in retrieved_chunks:
        answer_units.extend(split_answer_units(str(chunk["text"])))

    if not answer_units:
        return "I don't know"

    local_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    matrix = local_vectorizer.fit_transform([question] + answer_units).astype("float32").toarray()

    query_vector = matrix[0:1]
    unit_vectors = matrix[1:]

    query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True) + 1e-12
    unit_norms = np.linalg.norm(unit_vectors, axis=1, keepdims=True) + 1e-12
    similarities = (unit_vectors @ query_vector.T).reshape(-1) / (unit_norms.reshape(-1) * query_norm.reshape(-1)[0])

    best_indices = similarities.argsort()[::-1][:2]
    selected = []

    for idx in best_indices:
        if similarities[idx] >= 0.05:
            selected.append(answer_units[int(idx)])

    if not selected:
        return "I don't know"

    # Remove duplicate selected units while preserving order.
    final_units = []
    seen = set()
    for unit in selected:
        if unit not in seen:
            seen.add(unit)
            final_units.append(unit)

    return "\n\n".join(final_units)


def answer_question(rag_index: LocalRAGIndex, question: str) -> Tuple[str, List[Dict[str, str | float]]]:
    retrieved_chunks = retrieve_top_k(rag_index, question, k=3)
    answer = create_extractive_answer(question, retrieved_chunks)
    return answer, retrieved_chunks
