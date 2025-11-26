import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rag_chatbot import (
    build_rag_components,
    retrieve_transactions,
    simple_answer_from_context
)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# ------------------ LOAD DATA & COMPONENTS ------------------
DATA_PATH = Path("transactions.json")
components = build_rag_components(DATA_PATH)

# ------------------ MEMORY INITIALIZATION ------------------
if "last_question" not in st.session_state:
    st.session_state.last_question = None   # Most recent question
if "previous_question" not in st.session_state:
    st.session_state.previous_question = None  # Past question


# ------------------ UI HEADER ------------------
st.title("🛒 RAG-Powered Chatbot for Transactional Data")

# ------------------ MAIN INPUT ------------------
query = st.text_input("Ask a question:", "What is Amit's total spending?")

top_k = st.slider("Top K retrieval", 1, 5, 3)

col1, col2 = st.columns(2)

with col1:
    get_answer_button = st.button("Get Answer")

with col2:
    show_memory_button = st.button("Show my last question")


# ------------------ MAIN LOGIC ------------------
if get_answer_button:

    # ⭐ Save previous question BEFORE updating the new one
    st.session_state.previous_question = st.session_state.last_question

    # ⭐ Save new current question
    st.session_state.last_question = query

    # Retrieve top-k similar transactions
    retrieved = retrieve_transactions(
        query,
        components["vectorizer"],
        components["embeddings"],
        components["texts"],
        top_k=top_k
    )

    retrieved_texts = [r[1] for r in retrieved]

    st.subheader("Retrieved Context")
    for idx, text, score in retrieved:
        st.write(f"**Score:** {score:.3f} — {text}")

    # Generate final answer
    answer = simple_answer_from_context(retrieved_texts, query)

    st.subheader("Answer")
    st.success(answer)


# ------------------ MEMORY FEATURE ------------------
if show_memory_button:
    st.subheader("🧠 Last Question Memory")

    if st.session_state.previous_question:
        st.info(st.session_state.previous_question)
    else:
        st.warning("No previous question stored yet.")


# ------------------ BONUS: MONTHLY SPENDING CHART ------------------
df = components["df"].copy()
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)

spend = df.groupby("month")["amount"].sum()

st.subheader("📊 Spending Per Month")
fig, ax = plt.subplots()
ax.bar(spend.index, spend.values)
ax.set_xlabel("Month")
ax.set_ylabel("Total Spending")
st.pyplot(fig)
