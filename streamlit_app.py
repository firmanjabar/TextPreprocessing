# app.py
# Streamlit + NLTK + Stanza Text Processing App
# Fitur:
# - Upload dokumen (TXT/MD/DOCX) atau tempel teks
# - Pra-proses: casefolding, hapus angka/tanda baca, normalisasi spasi
# - Tokenisasi (NLTK), Stopwords (EN/ID), Lemmatization (Stanza)
# - POS tagging & NER (Stanza)
# - Frekuensi kata (Top-N) + unduh hasil (TXT/CSV)
# - UI modern: sidebar yang jelas, tab untuk hasil, progres bar

import io
import re
import time
from collections import Counter

import streamlit as st

# --- Optional imports only when needed ---
try:
    import docx  # python-docx
except Exception:
    docx = None  # agar app tetap jalan jika belum terpasang

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import stanza

# -------------------- CONFIG & THEME --------------------
st.set_page_config(
    page_title="Text Processing WebApp - Firman Jabar",
    page_icon="🧠",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* Header */
#root .block-container {padding-top: 1.5rem;}
h1, h2, h3 {font-weight: 700;}
/* Cards */
.card {
  border-radius: 16px;
  padding: 1rem 1.2rem;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.15);
}
/* Subtle divider */
.divider {height: 1px; background: linear-gradient(90deg, transparent, #888, transparent); margin: .8rem 0;}
/* Pills */
.pill {
  display:inline-block; padding:.2rem .6rem; border-radius:999px;
  border:1px solid rgba(127,127,127,.4); margin:.15rem; font-size:.85rem;
}
.small-muted {opacity:.7; font-size:.9rem}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- CACHING --------------------
@st.cache_resource(show_spinner=False)
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

@st.cache_resource(show_spinner=False)
def get_stopwords(lang_code):
    ensure_nltk()
    if lang_code == "id":
        try:
            return set(stopwords.words("indonesian"))
        except OSError:
            # Fallback jika korpus indonesian tidak tersedia di versi lama
            return set()
    else:
        return set(stopwords.words("english"))

@st.cache_resource(show_spinner=True)
def load_stanza_pipeline(lang_code: str, processors: str):
    """
    processors bisa berisi subset, mis:
      - "tokenize,lemma,pos,ner"
      - "tokenize,lemma"
    """
    try:
        stanza.Pipeline  # just to ensure import OK
    except Exception as e:
        st.error("Stanza belum terpasang. Jalankan: pip install stanza")
        raise e

    try:
        # coba buat pipeline; kalau model belum ada, download dulu
        return stanza.Pipeline(lang=lang_code, processors=processors, tokenize_no_ssplit=True, verbose=False)
    except Exception:
        stanza.download(lang_code, verbose=False)
        return stanza.Pipeline(lang=lang_code, processors=processors, tokenize_no_ssplit=True, verbose=False)

# -------------------- HELPERS --------------------
def read_file(uploaded):
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")
    elif name.endswith(".docx"):
        if docx is None:
            st.warning("File DOCX terdeteksi, namun paket `python-docx` tidak terpasang. Gunakan: `pip install python-docx` atau unggah TXT.")
            return ""
        f = io.BytesIO(data)
        doc = docx.Document(f)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        st.warning("Format tidak dikenali. Unggah .txt / .md / .docx")
        return ""

def clean_text(text, lowercase=True, remove_digits=True, remove_punct=True, normalize_space=True):
    if lowercase:
        text = text.lower()
    if remove_digits:
        text = re.sub(r"\d+", " ", text)
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        # catatan: \w akan mempertahankan karakter alfanumerik + underscore
    if normalize_space:
        text = re.sub(r"\s+", " ", text).strip()
    return text

def nltk_tokenize(text, lang_code):
    # gunakan punkt (bahasa inggris berfungsi umum untuk token dasar)
    ensure_nltk()
    try:
        tokens = word_tokenize(text, language="indonesian" if lang_code=="id" else "english")
    except LookupError:
        # fallback
        tokens = text.split()
    return tokens

def remove_stopwords(tokens, lang_code, extra_stopwords=None, keep_short_tokens=True):
    sw = get_stopwords(lang_code)
    if extra_stopwords:
        sw |= set(extra_stopwords)
    out = []
    for t in tokens:
        if t in sw:
            continue
        if keep_short_tokens and len(t) == 1:
            # boleh buang token 1 huruf seperti 'a'/'i' jika mau
            continue
        out.append(t)
    return out

def stanza_process(text, lang_code, do_lemma=True, do_pos=True, do_ner=True):
    processors = ["tokenize"]
    if do_lemma:
        processors.append("lemma")
    if do_pos:
        processors.append("pos")
    if do_ner:
        processors.append("ner")
    pipe = load_stanza_pipeline(lang_code, ",".join(processors))
    doc = pipe(text)
    # keluaran ringkas
    lemmas = []
    pos_tags = []
    entities = []
    for sent in doc.sentences:
        for w in sent.words:
            tok = w.text
            lem = w.lemma if hasattr(w, "lemma") else tok
            pos = w.upos if hasattr(w, "upos") else None
            lemmas.append(lem)
            if pos:
                pos_tags.append((tok, pos))
        if do_ner and hasattr(sent, "entities"):
            for ent in sent.entities:
                entities.append((ent.text, ent.type))
    return lemmas, pos_tags, entities

def to_csv_lines(rows, header=None):
    buf = io.StringIO()
    if header:
        buf.write(",".join(header) + "\n")
    for r in rows:
        if isinstance(r, (list, tuple)):
            buf.write(",".join(map(lambda x: str(x).replace(",", " "), r)) + "\n")
        else:
            buf.write(str(r) + "\n")
    return buf.getvalue().encode("utf-8")


# -------------------- UI: SIDEBAR --------------------
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.markdown(
    "<div class='small-muted'>Atur bahasa & langkah pra-proses. "
    "Gunakan Stanza untuk lemmatization, POS, dan NER.</div>",
    unsafe_allow_html=True,
)

lang = st.sidebar.selectbox("Bahasa dokumen", ["id (Indonesia)", "en (English)"])
lang_code = "id" if lang.startswith("id") else "en"

st.sidebar.subheader("Pra-proses")
lowercase = st.sidebar.checkbox("Casefolding (lowercase)", True)
remove_digits = st.sidebar.checkbox("Hapus angka", False)
remove_punct = st.sidebar.checkbox("Hapus tanda baca", True)
normalize_space = st.sidebar.checkbox("Normalisasi spasi", True)

st.sidebar.subheader("Tokenisasi & Stopwords (NLTK)")
use_stop = st.sidebar.checkbox("Buang stopwords", True)
extra_sw = st.sidebar.text_input("Tambahan stopwords (pisah koma)", value="")

st.sidebar.subheader("Stanza NLP")
do_lemma = st.sidebar.checkbox("Lemmatization", True)
do_pos   = st.sidebar.checkbox("POS Tagging", True)
do_ner   = st.sidebar.checkbox("Named Entity Recognition (NER)", True)

topn = st.sidebar.slider("Top-N Frekuensi Kata", min_value=10, max_value=100, value=30, step=5)

st.sidebar.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='small-muted'>Tip: Unduh model Stanza akan otomatis saat pertama kali dipakai (sekali saja).</div>",
    unsafe_allow_html=True,
)

# -------------------- UI: MAIN --------------------
st.title("🧠 Text Processing WebApp")
st.markdown(
    """
    <div class='card'>
      <b>Tujuan:</b> memudahkan pra-proses & analisis awal teks (tokenisasi, stopwords, lemmatization, POS, NER) dengan antarmuka yang simpel.<br>
      Unggah dokumen (TXT/MD/DOCX) atau tempel teks. Atur pipeline di sidebar.
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.2, 1])
with col1:
    uploaded = st.file_uploader("📄 Unggah dokumen (.txt, .md, .docx)", type=["txt", "md", "docx"])
with col2:
    st.write("")
    st.write("")
    paste_mode = st.toggle("📝 Mode tempel teks", value=False)

text = ""
if uploaded is not None:
    text = read_file(uploaded)

if paste_mode:
    sample_placeholder = "Tempel teks di sini... (atau biarkan kosong untuk contoh)"
    text = st.text_area("Input Teks", value=text or sample_placeholder, height=200)
    if text.strip() == sample_placeholder:
        text = ""

# contoh teks bila kosong
if not text:
    text = (
        "Ini adalah contoh teks berbahasa Indonesia untuk mendemonstrasikan aplikasi. "
        "Aplikasi ini melakukan pra-proses, tokenisasi, stopwords, lemmatization, POS tagging, dan NER. "
        "Anda bisa mengganti bahasa ke English untuk mencoba model Stanza yang berbeda."
    )

# -------------------- PROSES --------------------
st.markdown("### 🔧 Jalankan Pemrosesan")
run = st.button("🚀 Proses Teks")

if run:
    progress = st.progress(0, text="Memulai...")

    # 1) Preprocessing dasar
    t0 = time.time()
    cleaned = clean_text(
        text,
        lowercase=lowercase,
        remove_digits=remove_digits,
        remove_punct=remove_punct,
        normalize_space=normalize_space,
    )
    progress.progress(20, text="Pra-proses selesai")

    # 2) Tokenisasi & stopwords
    tokens = nltk_tokenize(cleaned, lang_code)
    if use_stop:
        extras = [w.strip() for w in extra_sw.split(",") if w.strip()]
        tokens = remove_stopwords(tokens, lang_code, extras, keep_short_tokens=True)
    progress.progress(45, text="Tokenisasi & Stopwords selesai")

    # 3) Lemmatization / POS / NER (Stanza)
    lemmas, pos_tags, ents = [], [], []
    if do_lemma or do_pos or do_ner:
        try:
            lemmas, pos_tags, ents = stanza_process(cleaned, lang_code, do_lemma, do_pos, do_ner)
        except Exception as e:
            st.error(f"Gagal menjalankan Stanza: {e}")
    progress.progress(75, text="Analisis Stanza selesai")

    # 4) Frekuensi kata
    counter = Counter(lemmas if (do_lemma and lemmas) else tokens)
    top_items = counter.most_common(topn)
    progress.progress(100, text="Selesai ✅")

    # -------------------- OUTPUT TABS --------------------
    st.markdown("### 📊 Hasil")
    tab1, tab2, tab3, tab4 = st.tabs(["Teks Bersih", "Tokens & Lemmas", "POS Tagging", "NER"])

    with tab1:
        st.text_area("Hasil Pra-proses", value=cleaned, height=200)
        st.download_button(
            "⬇️ Unduh Teks Bersih (.txt)",
            data=cleaned.encode("utf-8"),
            file_name="cleaned_text.txt",
            mime="text/plain",
        )

    with tab2:
        st.markdown("#### Frekuensi Kata (Top-N)")
        # tampilkan tabel sederhana
        if top_items:
            st.dataframe(
                {"token": [w for w, _ in top_items], "freq": [c for _, c in top_items]},
                use_container_width=True,
            )
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Daftar Tokens / Lemmas")
        colA, colB = st.columns(2)
        with colA:
            st.caption("Tokens (NLTK)")
            st.write(", ".join(tokens[:500]) + (" ..." if len(tokens) > 500 else ""))
        with colB:
            st.caption("Lemmas (Stanza)")
            st.write(", ".join(lemmas[:500]) + (" ..." if len(lemmas) > 500 else ""))

        csv_freq = to_csv_lines(top_items, header=["token", "freq"])
        st.download_button(
            "⬇️ Unduh Frekuensi Kata (.csv)",
            data=csv_freq,
            file_name="word_frequency.csv",
            mime="text/csv",
        )

    with tab3:
        st.markdown("#### POS Tagging (Stanza)")
        if pos_tags:
            st.dataframe({"token": [t for t, _ in pos_tags], "upos": [p for _, p in pos_tags]}, use_container_width=True)
            csv_pos = to_csv_lines(pos_tags, header=["token", "upos"])
            st.download_button(
                "⬇️ Unduh POS (.csv)", data=csv_pos, file_name="pos_tags.csv", mime="text/csv"
            )
        else:
            st.info("POS tidak tersedia (matikan/nyalakan opsi atau cek bahasa).")

    with tab4:
        st.markdown("#### Named Entities (Stanza)")
        if ents:
            st.dataframe({"entity": [e for e, _ in ents], "type": [t for _, t in ents]}, use_container_width=True)
            csv_ner = to_csv_lines(ents, header=["entity", "type"])
            st.download_button(
                "⬇️ Unduh NER (.csv)", data=csv_ner, file_name="named_entities.csv", mime="text/csv"
            )
        else:
            st.info("Tidak ada entitas terdeteksi (coba bahasa English untuk cakupan NER yang lebih luas).")

else:
    st.info("Atur opsi di sidebar, unggah/tempeI teks, lalu klik **🚀 Proses Teks** untuk mulai.")
