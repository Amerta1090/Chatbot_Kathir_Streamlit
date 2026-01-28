# app_streamlit_extreme.py (versi clean)
import streamlit as st
st.set_page_config(page_title="RAG Chat Interface", layout="wide")
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
DATA_DIR = "data"
BM25_PATH = os.path.join(DATA_DIR, "bm25_model.pkl")
DENSE_EMB_PATH = os.path.join(DATA_DIR, "dense_embeddings.npy")
EMB_INDEX_PATH = os.path.join(DATA_DIR, "embedding_index.csv")
CLEAN_DATASET_PATH = os.path.join(DATA_DIR, "clean_dataset.csv")
PROMPT_PATH = os.path.join(DATA_DIR, "prompt_rag.txt")

# HF Router config
HF_TOKEN = st.secrets.get("hf_token", None)
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"
HF_BASE_URL = "https://router.huggingface.co/v1"

# ----------------------------
# Surah mapping
# ----------------------------
SURAH_NAMES = [
    "Al-Fatihah", "Al-Baqarah", "Ali 'Imran", "An-Nisa'", "Al-Ma'idah", "Al-An'am", "Al-A'raf",
    "Al-Anfal", "At-Taubah", "Yunus", "Hud", "Yusuf", "Ar-Ra'd", "Ibrahim", "Al-Hijr", "An-Nahl",
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
SURAH_MAP = {i + 1: SURAH_NAMES[i] for i in range(len(SURAH_NAMES))}
NAME_TO_NUM = {v: k for k, v in SURAH_MAP.items()}

# ----------------------------
# Utility functions
# ----------------------------
def simple_tokenizer(text: str):
    return text.lower().split()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def min_max_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.0]*len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)

def get_document_text(doc_id, df):
    try:
        i = int(doc_id)
    except:
        return ""
    if 0 <= i < len(df):
        return df.iloc[i].get('text', "")
    return ""

# ----------------------------
# Load resources
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_resources():
    # BM25
    try:
        with open(BM25_PATH, 'rb') as f:
            bm25_model = pickle.load(f)
    except:
        bm25_model = None

    # Dense embeddings
    try:
        dense_embeddings = np.load(DENSE_EMB_PATH)
    except:
        dense_embeddings = np.zeros((1, 768))

    # Index & documents
    try:
        embedding_index = pd.read_csv(EMB_INDEX_PATH)
    except:
        embedding_index = pd.DataFrame()

    try:
        full_document_texts_df = pd.read_csv(CLEAN_DATASET_PATH)
    except:
        full_document_texts_df = pd.DataFrame()

    # Multilingual E5
    try:
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
        model.eval()
    except:
        tokenizer, model = None, None

    return bm25_model, dense_embeddings, embedding_index, full_document_texts_df, tokenizer, model

bm25_model, dense_embeddings, embedding_index, full_document_texts_df, tokenizer, model = load_resources()

# ----------------------------
# Hybrid search — filter before retrieval
# ----------------------------
def hybrid_search(query: str, filters: Dict[str, Any]=None, k: int=1) -> pd.DataFrame:
    # 1. Tentukan subset dokumen berdasarkan filter
    df_filtered = full_document_texts_df.copy()
    if filters:
        if 'sura' in filters and filters['sura']:
            df_filtered = df_filtered[df_filtered['sura'] == filters['sura']]
        if 'ayah' in filters and filters['ayah']:
            df_filtered = df_filtered[
                (df_filtered['tafsir_ayah_start'] <= filters['ayah']) &
                (df_filtered['tafsir_ayah_end'] >= filters['ayah'])
            ]
    if df_filtered.empty:
        # fallback: gunakan seluruh dataset
        df_filtered = full_document_texts_df.copy()
    df_indices = df_filtered.index.to_numpy()

    # 2. BM25
    if bm25_model:
        tokenized_q = simple_tokenizer(query)
        bm25_scores_all = bm25_model.get_scores(tokenized_q)
        bm25_scores = bm25_scores_all[df_indices]  # subset
    else:
        bm25_scores = np.zeros(len(df_indices))

    top_sparse_idx = np.argsort(bm25_scores)[::-1][:k]
    top_sparse_doc_ids = df_indices[top_sparse_idx]
    sparse_df = pd.DataFrame([
        {"doc_id": int(doc_id), "bm25_score": float(bm25_scores[i])}
        for i, doc_id in enumerate(top_sparse_doc_ids)
    ])

    # 3. Dense embeddings
    if tokenizer and model:
        q_inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            q_out = model(**q_inputs)
            q_emb = mean_pooling(q_out, q_inputs['attention_mask'])
        q_emb_np = q_emb.cpu().numpy()
        dense_scores_all = cosine_similarity(q_emb_np, dense_embeddings[df_indices])[0]
    else:
        dense_scores_all = np.zeros(len(df_indices))

    top_dense_idx = np.argsort(dense_scores_all)[::-1][:k]
    top_dense_doc_ids = df_indices[top_dense_idx]
    dense_df = pd.DataFrame([
        {"doc_id": int(doc_id), "dense_score": float(dense_scores_all[i])}
        for i, doc_id in enumerate(top_dense_doc_ids)
    ])

    # 4. Merge BM25 & dense, normalize, fused score
    merged = pd.merge(sparse_df, dense_df, on='doc_id', how='outer').fillna(0.0)
    merged['normalized_bm25_score'] = min_max_normalize(merged['bm25_score'])
    merged['normalized_dense_score'] = min_max_normalize(merged['dense_score'])
    merged['fused_score'] = 0.5 * merged['normalized_dense_score'] + 0.5 * merged['normalized_bm25_score']
    merged = merged.sort_values('fused_score', ascending=False).head(k)

    # 5. Merge dengan embedding_index (metadata)
    right_key = 'id' if 'id' in embedding_index.columns else 'doc_id'
    if not embedding_index.empty:
        merged = pd.merge(merged, embedding_index, left_on='doc_id', right_on=right_key, how='inner')

    # 6. Ambil document_text
    merged['document_text'] = merged['doc_id'].apply(lambda x: get_document_text(x, full_document_texts_df))

    return merged.sort_values('fused_score', ascending=False)



# ----------------------------
# HF client
# ----------------------------
def init_hf_client():
    if HF_TOKEN is None:
        return None
    try:
        return OpenAI(base_url=HF_BASE_URL, api_key=HF_TOKEN)
    except:
        return None

hf_client = init_hf_client()

def call_hf_generate(prompt: str):
    if hf_client is None:
        return "Error: HF client tidak terinisialisasi."
    try:
        completion = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # UBAH PARAMETER INI:
            temperature=0.6,  
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
# ----------------------------
# Generate answer
# ----------------------------
def generate_answer(query: str, filters: Dict[str, Any]=None):
    context_df = hybrid_search(query, filters=filters, k=1)
    context_pieces = []
    for i, row in context_df.iterrows():
        snippet = str(row.get('document_text', ""))[:800].replace("\n", " ")
        context_pieces.append(f"[{i+1}] {snippet}")
    context_text = "\n\n".join(context_pieces) if context_pieces else "Tidak ada konteks tersedia."

    if os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        prompt_template =  (
		'''
		Context: {context}
		Question: {query}

		Instruction: Answer in the same language as the question. Use only the information from the context. If not available, write "No information available". Be clear and detailed.

		'''
		)

    prompt = prompt_template.format(context=context_text, query=query)
    answer = call_hf_generate(prompt)
    return answer, context_df

# ----------------------------
# UI State
# ----------------------------
if "filter_open" not in st.session_state:
    st.session_state.filter_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# CSS
# ----------------------------
st.set_page_config(page_title="RAG Chat Interface", layout="wide")
st.markdown("""
<style>
.stChatMessage { padding: 8px 12px; border-radius: 10px; margin-bottom: 8px; }
.assistant-bubble { background: linear-gradient(90deg, rgba(22,8,56,0.65), rgba(30,12,60,0.6)); border-left: 3px solid #8a63ff; }
.user-bubble { background: rgba(255,255,255,0.03); border-left: 3px solid #17bebb; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Top bar + filter
# ----------------------------
col1, col2 = st.columns([8, 2])
with col1:
    st.title("RAG — Tafsir Ibn Kathir Assistant")
with col2:
    if st.button("Filter", key="toggle_filter"):
        st.session_state.filter_open = not st.session_state.filter_open

if st.session_state.filter_open:
    with st.expander("Filter — Surah & Ayah", expanded=True):
        surah_name_selected = st.selectbox("Pilih Surah (nama)", ["Semua"] + SURAH_NAMES, index=0)
        selected_sura = NAME_TO_NUM.get(surah_name_selected) if surah_name_selected != "Semua" else None
        ayah_input = st.text_input("Nomor Ayah (opsional)", value="")
        selected_ayah = int(ayah_input) if ayah_input.isdigit() else None
else:
    selected_sura = None
    selected_ayah = None

# ----------------------------
# Chat columns
# ----------------------------
if "awaiting_answer" not in st.session_state:
    st.session_state.awaiting_answer = False
    st.session_state.last_query = ""

left_col, right_col = st.columns([3,1])
with left_col:
    # Render chat
    for idx, (role, text, ctx_df) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(
                f"<div class='stChatMessage user-bubble'><b>You</b><div>{text}</div></div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='stChatMessage assistant-bubble'><b>Assistant</b><div>{text}</div></div>", 
                unsafe_allow_html=True
            )
            if ctx_df is not None and not ctx_df.empty:
                with st.expander("Lihat konteks sumber (klik untuk buka)", expanded=False):
                    cols_to_show = [c for c in ['document_text'] if c in ctx_df.columns]
                    st.dataframe(ctx_df[cols_to_show].head(10))

    # Chat input
    user_query = st.chat_input("Tanyakan tentang tafsir — gunakan bahasa Indonesia atau campuran")
    if user_query:
        st.session_state.chat_history.append(("user", user_query, None))
        st.session_state.last_query = user_query
        st.session_state.awaiting_answer = True

# Generate assistant answer if flagged
if st.session_state.awaiting_answer:
    filters = {"sura": selected_sura, "ayah": selected_ayah}
    with st.spinner("Mencari dan menghasilkan jawaban..."):
        answer, ctx_df = generate_answer(st.session_state.last_query, filters=filters)
        st.session_state.chat_history.append(("assistant", answer, ctx_df))
    st.session_state.awaiting_answer = False
    st.rerun()
    # No need for experimental_rerun; Streamlit auto reruns after input


with right_col:
    st.header("Controls")
    if st.button("Clear chat"):
        st.session_state.chat_history = []
    st.write("Filter state:", "Open" if st.session_state.filter_open else "Closed")
    st.write("Selected surah:", SURAH_MAP.get(selected_sura, "Semua"))
    st.write("Selected ayah:", selected_ayah if selected_ayah else "Semua")

st.markdown("---")
st.markdown("Antarmuka ini memprioritaskan percakapan. Jika Anda ingin melihat konteks sumber, buka expander pada jawaban assistant.")
