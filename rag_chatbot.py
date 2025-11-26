import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- 1. Load & Preprocess ----------
DATA_PATH = Path("transactions.json")

def load_transactions(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def tx_to_text(row: pd.Series) -> str:
    return f"On {row['date']}, {row['customer']} purchased a {row['product']} for {row['amount']}."


# ---------- 2. Create embeddings (TF-IDF) ----------
def build_embeddings(texts: List[str]):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


# ---------- 3. Retriever ----------
def retrieve_transactions(query: str, vectorizer, embeddings_matrix, texts: List[str], top_k: int = 3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, embeddings_matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(int(i), texts[i], float(sims[i])) for i in top_idx]


# ---------- 4. Simple LLM simulation ----------
def compose_prompt(retrieved_texts: List[str], question: str) -> str:
    context = "\n".join(retrieved_texts)
    prompt = (
        "You are a helpful assistant that answers questions about retail transactions.\n"
        "Use ONLY this context. If the answer is not present, reply: 'I don't know'.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    return prompt


def simple_answer_from_context(retrieved_texts: List[str], question: str) -> str:
    q = question.lower()

    import re

    # simple customer detection
    customers = ["amit", "riya", "karan"]
    found_customer = None
    for c in customers:
        if c in q:
            found_customer = c.capitalize()
            break

    # total spend
    if "total" in q and ("spend" in q or "spent" in q):
        total = 0
        found = False
        for t in retrieved_texts:
            if found_customer and found_customer.lower() not in t.lower():
                continue
            m = re.search(r"for\s+([0-9]+)", t)
            if m:
                found = True
                total += int(m.group(1))
        return f"{found_customer} spent a total of {total}." if found else "I don't know."

    # purchase history
    if "purchase history" in q or ("list" in q and "transaction" in q):
        items = []
        for t in retrieved_texts:
            if found_customer and found_customer.lower() not in t.lower():
                continue
            items.append(t)
        if items:
            return " ; ".join(items)
        return "I don't know."

    # average order amount
    if "average" in q and ("order" in q or "amount" in q):
        vals = []
        for t in retrieved_texts:
            m = re.search(r"for\s+([0-9]+)", t)
            if m:
                vals.append(int(m.group(1)))
        if vals:
            return f"The average order amount is {round(sum(vals)/len(vals), 2)}."
        return "I don't know."

    # most purchased product
    if "most" in q and ("product" in q or "often" in q):
        from collections import Counter
        prods = []
        for t in retrieved_texts:
            m = re.search(r"purchased a\s+([A-Za-z ]+)\s+for", t)
            if m:
                prods.append(m.group(1).strip())
        if prods:
            p, c = Counter(prods).most_common(1)[0]
            return f"The most purchased product is {p} ({c} times)."
        return "I don't know."

    return "I don't know."


# ---------- 5. Main RAG system ----------
def build_rag_components(path: Path):
    df = load_transactions(path)
    df["text"] = df.apply(tx_to_text, axis=1)
    texts = df["text"].tolist()

    vectorizer, embeddings = build_embeddings(texts)

    return {
        "df": df,
        "texts": texts,
        "vectorizer": vectorizer,
        "embeddings": embeddings
    }


def chat_loop(components):
    print("RAG Chatbot started. Type 'exit' to stop.")

    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            break

        retrieved = retrieve_transactions(
            q,
            components["vectorizer"],
            components["embeddings"],
            components["texts"],
            top_k=3
        )

        retrieved_texts = [r[1] for r in retrieved]

        answer = simple_answer_from_context(retrieved_texts, q)
        print("Bot:", answer)


if __name__ == "__main__":
    comps = build_rag_components(DATA_PATH)
    chat_loop(comps)
