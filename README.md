

# Menjembatani Jurang Pengetahuan: Membawa Tafsir Klasik ke Era Digital dengan AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://chatbotibnukathir.streamlit.app/)

Repositori ini berisi implementasi sistem **Retrieval-Augmented Generation (RAG)** yang dirancang khusus untuk meningkatkan aksesibilitas terhadap **Tafsir Ibn Kathir**. Proyek ini menggunakan pendekatan **Hybrid Sparse-Dense Retrieval** untuk menghasilkan jawaban keagamaan yang akurat, relevan, dan setia pada sumber aslinya.

## 📌 Latar Belakang
Pemahaman terhadap tafsir Al-Qur’an merupakan kebutuhan esensial, namun akses terhadapnya menghadapi hambatan fundamental.
*   **Krisis Aksesibilitas:** Sekitar **82,5% non-penutur Arab** kesulitan memahami makna Al-Qur’an karena faktor bahasa dan keterbatasan perangkat interpretatif.
*   **Volume Luar Biasa:** Tafsir Ibn Kathir edisi ringkas terdiri dari **10 jilid dengan total lebih dari 6.500 halaman**, yang menjadi tantangan bagi pembaca modern.
*   **Tantangan Literasi:** Tren global menunjukkan penurunan kemampuan pemahaman bacaan (*reading comprehension*) di tingkat nasional dan internasional.

## 🚀 Fitur Utama
*   **Hybrid Sparse-Dense Retrieval:** Menggabungkan presisi leksikal **BM25** dengan jangkauan semantik **E5 Large Multilingual** untuk relevansi maksimal.
*   **Grounded Generation:** Menggunakan model **LLaMA 3.2 3B** yang dioptimalkan untuk menghasilkan jawaban berbasis fakta dari dokumen tafsir, guna meminimalkan risiko halusinasi.
*   **Multilingual Support:** Mampu memproses *query* dalam berbagai bahasa, termasuk bahasa Indonesia dan Inggris, dengan dukungan teks asli bahasa Arab.
*   **Penyaringan Metadata:** Fitur penyaringan pencarian berdasarkan surah dan ayat untuk meningkatkan presisi.
*   **Transparansi Sumber:** Menampilkan potongan teks asli yang digunakan AI sebagai rujukan jawaban.

## 🏗️ Arsitektur Sistem
Sistem ini bekerja melalui *pipeline* **end-to-end** sebagai berikut:
1.  **User Query:** Pengguna memasukkan pertanyaan terkait tafsir.
2.  **Hybrid Retrieval Engine:** Pencarian paralel menggunakan **Inverted Index (BM25)** dan **Vector DB (E5)**.
3.  **Score Fusion:** Penggabungan skor menggunakan **Reciprocal Rank Fusion** atau *alpha-weighted fusion* untuk mendapatkan konteks terbaik.
4.  **Context Augmentation:** Dokumen peringkat teratas digabungkan dengan kueri asli menjadi *prompt* yang kaya konteks.
5.  **Generation:** Model **LLaMA 3.2 3B** menghasilkan jawaban akhir berdasarkan konteks yang diberikan.

## 📊 Performa Sistem
Berdasarkan hasil pengujian eksperimental:
*   **Kinerja Retrieval:** Metode hibrida mencapai keseimbangan optimal dengan skor **nDCG@10 sebesar 0,94 (Leksikal)** dan **0,99 (Semantik)**.
*   **Kualitas Jawaban:** Skor **Context Relevance mencapai 0,972**, menunjukkan kesetiaan yang sangat tinggi terhadap teks sumber.
*   **Efisiensi:** Rata-rata waktu respons *end-to-end* adalah **~5,93 detik**, sangat kompetitif untuk dijalankan pada spesifikasi perangkat keras terbatas tanpa GPU.

## 🛠️ Alat dan Teknologi
*   **Bahasa Pemrograman:** Python 3.11+.
*   **Framework UI:** Streamlit.
*   **Model Embedding:** `intfloat/e5-large-multilingual`.
*   **Model Generatif:** `LLaMA 3.2 3B` via HuggingFace API.
*   **Library Utama:** `pandas`, `rank_bm25`, `sentence-transformers`, `torch`, `transformers`.

## 🔗 Tautan Penting
*   **Live Demo:** [Tafsir Ibn Kathir Assistant](https://chatbotibnukathir.streamlit.app/)
*   **Dataset:** Bersumber dari basis data **Al-Qur'an GreenTech** yang telah melalui proses prapemrosesan menjadi 1.896 entri bersih.

## 📝 Lisensi & Sitasi
Proyek ini dikembangkan untuk kepentingan akademik dan penelitian.
*   **Referensi Utama:** Penelitian didasarkan pada pengembangan sistem RAG untuk domain keagamaan spesifik.
*   **Data:** Tafsir Ibn Kathir (Abridged Version).

**Kontribusi:**
Saran dan umpan balik sangat dihargai untuk pengembangan sistem yang lebih komprehensif di masa depan.
