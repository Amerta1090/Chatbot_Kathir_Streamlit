import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any

# ----------------------------
# CONFIG PATH
# ----------------------------
DATA_DIR = "../data"
BM25_PATH = os.path.join(DATA_DIR, "bm25_model.pkl")
DENSE_EMB_PATH = os.path.join(DATA_DIR, "dense_embeddings.npy")
EMB_INDEX_PATH = os.path.join(DATA_DIR, "embedding_index.csv")
CLEAN_DATASET_PATH = os.path.join(DATA_DIR, "clean_dataset.csv")
PROMPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompt_rag.txt")

# ----------------------------
# OpenRouter LLM config
# ----------------------------
OPENROUTER_TOKEN = st.secrets.get("openrouter_api_key", None)
OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ----------------------------
# Surah mapping
# ----------------------------
SURAH_NAMES = [
    "Al-Fatihah", "Al-Baqarah", "Ali 'Imran", "An-Nisa'", "Al-Ma'idah", "Al-An'am", "Al-A'raf",
    "Al-Anfal", "At-Tahbah", "Yunus", "Hud", "Yusuf", "Ar-Ra'd", "Ibrahim", "Al-Hijr", "An-Nahl",
    "Al-Isra'", "Al-Kahf", "Maryam", "Ta-Ha", "Al-Anbiya'", "Al-Hajj", "Al-Mu'minun", "An-Nur",
    "Al-Furqan", "Ash-Shu'ara'", "An-Naml", "Al-Qasas", "Al-Ankabut", "Ar-Rum", "Luqman", "As-Sajdah",
    "Al-Ahzab", "Saba'", "Fatir", "Ya-Sin", "As-Saffat", "Sad", "Az-Zumar", "Ghafir (Al-Mu'min)",
    "Fussilat", "Ash-Shura", "Az-Zukhruf", "Ad-Dukhan", "Al-Jathiyah", "Al-Ahqaf", "Muhammad",
    "Al-Fath", "Al-Hujurat", "Qaf", "Adh-Dhariyat", "At-Tur", "An-Najm", "Al-Qamar", "Ar-Rahman",
    "Al-Waqi'ah", "Al-Hadid", "Al-Mujadila", "Al-Hashr", "Al-Mumtahanah", "As-Saff", "Al-Jumu'ah",
    "Al-Munafiqun", "At-Taghabun", "At-Talaq", "At-Tahrim", "Al-Mulk", "Al-Qalam", "Al-Haqqah",
    "Al-Ma'arij", "Nuh", "Al-Jinn", "Al-Muzzammil", "Al-Muddaththir", "Al-Qiyamah", "Al-Insan",
    "Al-Mursalat", "An-Naba'", "An-Nazi'at", "Abasa", "At-Takwir", "Al-Infitar", "Al-Mutaffifin",
    "Al-Inshiqaq", "Al-Buruj", "At-Tariq", "Al-A'la", "Al-Ghashiyah", "Al-Fajr", "Al-Balad",
    "Ash-Shams", "Al-Lail", "Ad-Duha", "Ash-Sharh", "At-Tin", "Al-'Alaq", "Al-Qadr", "Al-Bayyinah",
    "Az-Zalzalah", "Al-'Adiyat", "Al-Qari'ah", "At-Takathur", "Al-Asr", "Al-Humazah", "Al-Fil",
    "Quraysh", "Al-Ma'un", "Al-Kawthar", "Al-Kafirun", "An-Nasr", "Al-Masad", "Al-Ikhlas",
    "Al-Falaq", "An-Nas"
]
SURAH_OPTIONS = ["Semua Surah"] + SURAH_NAMES
NAME_TO_NUM = {name: i + 1 for i, name in enumerate(SURAH_NAMES)}

# ----------------------------
# Utility functions
# ----------------------------
def simple_tokenizer(text: str):
    return text.lower().split()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def min_max_normalize(series: pd.Series) -> pd.Series:
    if series.empty or series.max() == series.min():
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def get_document_text(doc_id, df):
    try:
        i = int(doc_id)
        if 0 <= i < len(df):
            return df.iloc[i].get("text", "")
    except:
        pass
    return ""

# ----------------------------
# Load resources
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_resources():
    try:
        with open(BM25_PATH, "rb") as f:
            bm25_model = pickle.load(f)
    except:
        bm25_model = None

    try:
        dense_embeddings = np.load(DENSE_EMB_PATH)
    except:
        dense_embeddings = np.zeros((1, 768))

    try:
        embedding_index = pd.read_csv(EMB_INDEX_PATH)
    except:
        embedding_index = pd.DataFrame()

    try:
        full_document_texts_df = pd.read_csv(CLEAN_DATASET_PATH)
    except:
        full_document_texts_df = pd.DataFrame()

    try:
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
        model.eval()
    except:
        tokenizer, model = None, None

    return bm25_model, dense_embeddings, embedding_index, full_document_texts_df, tokenizer, model

bm25_model, dense_embeddings, embedding_index, full_document_texts_df, tokenizer, model = load_resources()

# ----------------------------
# Hybrid search with Metadata Filtering
# ----------------------------
def hybrid_search(query: str, filters: Dict[str, Any] = None, k: int = 1) -> pd.DataFrame:
    # 1. Tentukan subset dokumen berdasarkan filter
    df_filtered = full_document_texts_df.copy()
    if filters:
        if filters.get("sura"):
            df_filtered = df_filtered[df_filtered["sura"] == filters["sura"]]
        if filters.get("ayah"):
            df_filtered = df_filtered[
                (df_filtered["tafsir_ayah_start"] <= filters["ayah"]) &
                (df_filtered["tafsir_ayah_end"] >= filters["ayah"])
            ]
    
    # Jika hasil filter kosong, fallback ke semua (atau beri peringatan)
    if df_filtered.empty:
        df_filtered = full_document_texts_df.copy()
    
    df_indices = df_filtered.index.to_numpy()

    # 2. BM25 Search pada subset
    if bm25_model:
        tokenized_q = simple_tokenizer(query)
        bm25_scores_all = bm25_model.get_scores(tokenized_q)
        bm25_scores = bm25_scores_all[df_indices]
    else:
        bm25_scores = np.zeros(len(df_indices))

    top_sparse_idx = np.argsort(bm25_scores)[::-1][:k]
    sparse_df = pd.DataFrame({
        "doc_id": df_indices[top_sparse_idx].astype(int),
        "bm25_score": bm25_scores[top_sparse_idx].astype(float)
    })

    # 3. Dense Search pada subset
    if tokenizer and model and len(df_indices) > 0:
        q_inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            q_out = model(**q_inputs)
            q_emb = mean_pooling(q_out, q_inputs['attention_mask'])
        q_emb_np = q_emb.cpu().numpy()
        dense_scores_subset = cosine_similarity(q_emb_np, dense_embeddings[df_indices])[0]
    else:
        dense_scores_subset = np.zeros(len(df_indices))

    top_dense_idx = np.argsort(dense_scores_subset)[::-1][:k]
    dense_df = pd.DataFrame({
        "doc_id": df_indices[top_dense_idx].astype(int),
        "dense_score": dense_scores_subset[top_dense_idx].astype(float)
    })

    # 4. Merge, Normalize, and Fuse
    merged = pd.merge(sparse_df, dense_df, on="doc_id", how="outer").fillna(0.0)
    merged["normalized_bm25_score"] = min_max_normalize(merged["bm25_score"])
    merged["normalized_dense_score"] = min_max_normalize(merged["dense_score"])
    merged["fused_score"] = 0.5 * merged["normalized_bm25_score"] + 0.5 * merged["normalized_dense_score"]

    # 5. Join dengan metadata jika tersedia
    right_key = 'id' if 'id' in embedding_index.columns else 'doc_id'
    if not embedding_index.empty:
        merged = pd.merge(merged, embedding_index, left_on='doc_id', right_on=right_key, how='inner')

    # 6. Ambil teks dokumen
    merged["document_text"] = merged["doc_id"].apply(lambda x: get_document_text(x, full_document_texts_df))
    
    return merged.sort_values("fused_score", ascending=False).head(k)

# ----------------------------
# LLM Logic
# ----------------------------
def init_llm_client():
    if not OPENROUTER_TOKEN: return None
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_TOKEN)

llm_client = init_llm_client()

def call_llm_generate(prompt: str) -> str:
    if not llm_client: return "Error: API Key tidak ditemukan."
    try:
        completion = llm_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def generate_answer(query: str, filters: Dict[str, Any] = None):
    ctx_df = hybrid_search(query, filters=filters, k=1)
    
    if not ctx_df.empty:
        context = ctx_df.iloc[0]["document_text"]
        # Jika filter aktif, beri tahu LLM Surah/Ayah mana ini
        meta_info = f" (Surah: {ctx_df.iloc[0].get('sura', '')}, Ayah: {ctx_df.iloc[0].get('tafsir_ayah_start', '')})"
    else:
        context = "Tidak ada konteks ditemukan."
        meta_info = ""

    if os.path.exists(PROMPT_PATH):
        prompt_template = open(PROMPT_PATH, encoding="utf-8").read()
    else:
        prompt_template = (
            "Konteks Tafsir:\n{context}\n\n"
            "Pertanyaan:\n{query}\n\n"
            "Instruksi:\nJawablah dengan bahasa Indonesia yang baik hanya berdasarkan konteks di atas."
        )

    prompt = prompt_template.format(context=context, query=query)
    answer = call_llm_generate(prompt)
    return answer, ctx_df

# ----------------------------
# UI STREAMLIT
# ----------------------------
st.set_page_config(page_title="RAG Tafsir Ibn Kathir", layout="wide")

# SIDEBAR FILTER
st.sidebar.header("Filter Pencarian")
selected_surah_name = st.sidebar.selectbox("Pilih Surah", SURAH_OPTIONS)
selected_ayah = st.sidebar.number_input("Nomor Ayah (0 untuk semua)", min_value=0, step=1, value=0)

active_filters = {}
if selected_surah_name != "Semua Surah":
    active_filters["sura"] = NAME_TO_NUM[selected_surah_name]
if selected_ayah > 0:
    active_filters["ayah"] = selected_ayah

st.title("📖 RAG — Tafsir Ibn Kathir")
st.caption(f"Status Filter: **{selected_surah_name}** | Ayah: **{selected_ayah if selected_ayah > 0 else 'Semua'}**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Layout Chat
query = st.chat_input("Tanyakan tafsir (contoh: 'Apa kandungan utama ayat ini?')...")

if query:
    st.session_state.chat_history.append(("user", query, None))
    with st.spinner("Mencari di kitab tafsir..."):
        answer, ctx_df = generate_answer(query, filters=active_filters)
    st.session_state.chat_history.append(("assistant", answer, ctx_df))

# Tampilkan Chat
for role, text, ctx_df in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)
        if role == "assistant" and ctx_df is not None and not ctx_df.empty:
            with st.expander("📚 Sumber Konteks"):
                st.write(f"**Surah {ctx_df.iloc[0].get('sura')} | Ayah {ctx_df.iloc[0].get('tafsir_ayah_start')}**")
                st.write(ctx_df.iloc[0]["document_text"])
