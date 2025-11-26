# 🛒 RAG-Powered Chatbot for Transactional Data

*A Retrieval-Augmented Generation (RAG) chatbot built using Python and Streamlit.*

---

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on **customer transaction data**.
It simulates a real retail analytics assistant capable of:

* Showing purchase history
* Calculating total spending
* Finding frequently purchased products
* Summarizing month-wise spending

The system uses **TF-IDF embeddings**, **cosine similarity**, and **context-based answer generation**.

---

## 🚀 Features

### 🔍 Retrieval-Augmented Generation (RAG)

* Converts transaction records into natural language
* Builds embeddings for semantic search
* Retrieves top-k most relevant transactions
* Generates answers using context only

### 🧠 Memory Feature

* “Show my last question” button
* Stores the user’s previous question using `st.session_state`

### 🌐 Streamlit Web App

* Clean UI for asking questions
* Displays retrieved context
* Displays the chatbot's final answer
* Monthly spending visualization (bar chart)

### 📊 Analytics

* Spending per month
* Automatically parsed from transaction dates

---

## 📂 Project Structure

```
├── app_streamlit.py        # Streamlit UI (frontend)
├── rag_chatbot.py          # Backend RAG logic
├── transactions.json       # Sample dataset
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🧠 How RAG Works Here

### **1. Retrieve**

User question → TF-IDF → embedding → cosine similarity → top-k matches

### **2. Augment**

Combine retrieved context + user query

### **3. Generate**

Rule-based answer:

* sum amounts
* list purchases
* extract most common products
* filter by month

This follows the standard **retrieve → augment → generate** RAG pattern.

---

## 📘 Dataset (transactions.json)

```json
[
  {"id": 1, "customer": "Amit", "product": "Laptop", "amount": 55000, "date": "2024-01-12"},
  {"id": 2, "customer": "Amit", "product": "Mouse", "amount": 700, "date": "2024-02-15"},
  {"id": 3, "customer": "Riya", "product": "Mobile", "amount": 30000, "date": "2024-01-05"},
  {"id": 4, "customer": "Riya", "product": "Earbuds", "amount": 1500, "date": "2024-02-20"},
  {"id": 5, "customer": "Karan", "product": "Keyboard", "amount": 1200, "date": "2024-03-01"}
]
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit app

```
streamlit run app_streamlit.py
```

### 3️⃣ Open in browser:

```
http://localhost:8501
```

---

## 🧪 Example Questions to Try

* *What is Amit’s total spending?*
* *Show me Riya’s purchase history.*
* *List all transactions for February.*
* *Which product was purchased most often?*

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* Scikit-learn (TF-IDF Vectorizer)
* Matplotlib
* Cosine Similarity Retrieval
* Session-State Memory

---

## 🌟 Future Enhancements

* Replace rule-based logic with GPT/Claude-based LLM responses
* Add Sentence-Transformer semantic embeddings
* Add chat history
* Deploy to Streamlit Cloud or Render

---

## ✨ Author

**Hari Om**
B.Tech CSE (Data Science)
AI/ML | Data Science | RAG Systems | Streamlit Applications
