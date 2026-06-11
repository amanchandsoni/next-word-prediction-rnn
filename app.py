import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

@st.cache_resource
def load_artifacts():
    with open('nextword_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('nextword_maxlen.pkl', 'rb') as f:
        max_len = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1  # 8725

    model = Sequential([
        Embedding(input_dim=10000, output_dim=50, input_length=max_len-1),
        LSTM(128),
        Dropout(0.2),
        Dense(10000, activation='softmax')
    ])
    model.build(input_shape=(None, max_len-1))
    model.load_weights('nextword_weights.weights.h5')
    return model, tokenizer, max_len, vocab_size

def generate_text(seed_text, num_words, model, tokenizer, max_len, temperature=0.8):
    result = seed_text
    current = seed_text.lower()
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([current])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        predicted = np.random.choice(len(predictions), p=predictions)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        current += " " + output_word
        result += " " + output_word
    return result

st.set_page_config(page_title="Next Word Predictor", page_icon="✨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

*, html, body { margin:0; padding:0; box-sizing:border-box; }
.stApp { background:#06060e; font-family:'DM Sans',sans-serif; }

/* ANIMATED BG */
.orb { position:fixed; border-radius:50%; filter:blur(100px); pointer-events:none; z-index:0; }
.orb1 { width:700px; height:700px; background:radial-gradient(circle,#6d28d9,transparent); top:-250px; left:-200px; opacity:0.14; animation:o1 10s ease-in-out infinite alternate; }
.orb2 { width:500px; height:500px; background:radial-gradient(circle,#0284c7,transparent); bottom:-150px; right:-150px; opacity:0.1; animation:o2 13s ease-in-out infinite alternate; }
.orb3 { width:350px; height:350px; background:radial-gradient(circle,#db2777,transparent); top:45%; left:42%; opacity:0.07; animation:o3 9s ease-in-out infinite alternate; }
@keyframes o1 { to{transform:translate(50px,40px) scale(1.1)} }
@keyframes o2 { to{transform:translate(-40px,-50px) scale(1.08)} }
@keyframes o3 { to{transform:translate(30px,-30px) scale(1.05)} }

/* HERO */
.hero { text-align:center; padding:3.5rem 1rem 2rem; position:relative; z-index:1; }
.badge {
    display:inline-block;
    background:linear-gradient(135deg,rgba(109,40,217,0.18),rgba(2,132,199,0.12));
    border:1px solid rgba(109,40,217,0.35); border-radius:100px;
    padding:0.45rem 1.5rem; font-size:0.68rem; letter-spacing:4px;
    text-transform:uppercase; color:#a78bfa; margin-bottom:1.5rem;
}
.hero-title {
    font-family:'Playfair Display',serif; font-size:5.5rem; font-weight:900;
    color:#fff; line-height:1; margin-bottom:1rem;
    text-shadow:0 0 120px rgba(109,40,217,0.4);
}
.gt {
    background:linear-gradient(135deg,#a78bfa 0%,#38bdf8 50%,#f472b6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub { color:#374151; font-size:1rem; font-weight:300; letter-spacing:0.3px; }

/* STATS */
.stats { display:flex; gap:1rem; justify-content:center; max-width:1050px; margin:2rem auto; position:relative; z-index:1; }
.sc {
    flex:1; background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.065);
    border-radius:22px; padding:1.4rem; text-align:center; transition:all 0.3s; position:relative; overflow:hidden;
}
.sc::after { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,#7c3aed,transparent); opacity:0; transition:0.3s; }
.sc:hover::after { opacity:1; }
.sc:hover { border-color:rgba(109,40,217,0.3); transform:translateY(-5px); box-shadow:0 20px 50px rgba(0,0,0,0.4); }
.sv { font-family:'Playfair Display',serif; font-size:1.9rem; font-weight:700; color:#a78bfa; }
.sk { font-size:0.6rem; letter-spacing:2px; text-transform:uppercase; color:#1f2937; margin-top:0.3rem; }

/* BUTTONS */
.stButton > button {
    font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
    font-size:0.92rem !important; border-radius:14px !important;
    padding:0.82rem 1.5rem !important; width:100% !important;
    border:none !important; transition:all 0.3s ease !important; letter-spacing:0.3px !important;
    cursor:pointer !important;
}
/* Generate - first col */
div[data-testid="column"]:first-child .stButton > button {
    background:linear-gradient(135deg,#6d28d9 0%,#4f46e5 50%,#0284c7 100%) !important;
    color:white !important;
    box-shadow:0 4px 24px rgba(109,40,217,0.5),inset 0 1px 0 rgba(255,255,255,0.15) !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    transform:translateY(-3px) !important;
    box-shadow:0 10px 35px rgba(109,40,217,0.7) !important;
}
/* Clear - second col */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background:rgba(255,255,255,0.04) !important; color:#6b7280 !important;
    border:1px solid rgba(255,255,255,0.09) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background:rgba(239,68,68,0.12) !important; color:#ef4444 !important;
    border-color:rgba(239,68,68,0.3) !important;
}
/* Quick seed buttons */
div[data-testid="column"]:nth-child(3) .stButton > button,
div[data-testid="column"]:nth-child(4) .stButton > button {
    background:rgba(109,40,217,0.08) !important; color:#c4b5fd !important;
    border:1px solid rgba(109,40,217,0.2) !important; font-size:0.82rem !important;
}
div[data-testid="column"]:nth-child(3) .stButton > button:hover,
div[data-testid="column"]:nth-child(4) .stButton > button:hover {
    background:rgba(109,40,217,0.2) !important; border-color:rgba(109,40,217,0.5) !important;
    transform:translateY(-2px) !important;
}

/* Text input */
.stTextInput > div > div > input {
    background:rgba(255,255,255,0.03) !important; border:1px solid rgba(255,255,255,0.08) !important;
    border-radius:14px !important; color:#e2e8f0 !important;
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important; padding:0.8rem 1rem !important;
    transition:all 0.3s !important;
}
.stTextInput > div > div > input:focus {
    border-color:#7c3aed !important; box-shadow:0 0 0 4px rgba(109,40,217,0.12) !important;
}

/* Slider */
.stSlider > div > div > div > div { background:linear-gradient(90deg,#6d28d9,#38bdf8) !important; }

/* OUTPUT */
.output-box {
    background:linear-gradient(135deg,rgba(109,40,217,0.07),rgba(2,132,199,0.04));
    border:1px solid rgba(109,40,217,0.22); border-radius:26px; padding:2rem;
    position:relative; overflow:hidden; animation:fadeup 0.5s ease both;
}
.output-box::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#6d28d9,#38bdf8,#f472b6);
}
@keyframes fadeup { from{transform:translateY(15px);opacity:0} to{transform:translateY(0);opacity:1} }
.out-label { font-size:0.6rem; letter-spacing:3px; text-transform:uppercase; color:#7c3aed; margin-bottom:1.2rem; }
.out-text {
    font-family:'Playfair Display',serif; font-size:1.3rem; line-height:1.9; color:#e2e8f0;
    font-style:italic;
}
.seed-hl { color:#a78bfa; font-weight:700; font-style:normal; }

/* Word chips */
.chips { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:1.2rem; }
.chip {
    background:rgba(109,40,217,0.1); border:1px solid rgba(109,40,217,0.2);
    border-radius:100px; padding:0.22rem 0.8rem; font-size:0.78rem; color:#c4b5fd;
    animation:chipop 0.3s ease both;
}
@keyframes chipop { from{transform:scale(0.7);opacity:0} to{transform:scale(1);opacity:1} }

/* PANEL LABEL */
.pl { font-size:0.6rem; letter-spacing:3px; text-transform:uppercase; color:#6d28d9; margin-bottom:0.9rem; display:flex; align-items:center; gap:0.4rem; }
.pl::before { content:'▸'; font-size:0.7rem; }

/* COPY BOX */
.stTextArea textarea {
    background:rgba(255,255,255,0.02) !important; border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:14px !important; color:#9ca3af !important;
    font-family:'DM Sans',sans-serif !important; font-size:0.88rem !important;
}

/* EMPTY STATE */
.empty { text-align:center; padding:5rem 2rem; }
.empty-icon { font-size:3.5rem; opacity:0.05; }
.empty-txt { color:#111827; font-size:0.82rem; letter-spacing:1px; margin-top:1rem; line-height:1.8; }

/* SIDEBAR */
section[data-testid="stSidebar"] { background:#040410 !important; border-right:1px solid rgba(255,255,255,0.04) !important; }
section[data-testid="stSidebar"] .stButton > button {
    background:linear-gradient(135deg,#6d28d9,#4f46e5) !important;
    color:white !important; box-shadow:0 4px 16px rgba(109,40,217,0.4) !important;
}
#MainMenu, footer, header { visibility:hidden; }
div[data-testid="stMarkdownContainer"] p { color:#6b7280; }

/* Expander */
details { background:rgba(255,255,255,0.02) !important; border:1px solid rgba(255,255,255,0.07) !important; border-radius:14px !important; }
</style>

<div class='orb orb1'></div>
<div class='orb orb2'></div>
<div class='orb orb3'></div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.5rem'>
        <div style='font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;color:#f0eeff;margin-bottom:0.3rem'>🧠 Model Specs</div>
        <div style='color:#374151;font-size:0.73rem;margin-bottom:1.5rem;letter-spacing:0.3px'>LSTM Architecture Details</div>
    </div>
    """, unsafe_allow_html=True)

    specs = [
        ("🏗️ Architecture", "LSTM Sequential"),
        ("📚 Vocab Size", "8,725 words"),
        ("🔢 Embedding Dim", "50 dimensions"),
        ("🧬 LSTM Units", "128 units"),
        ("🎯 Dropout", "0.2 (20%)"),
        ("📏 Max Sequence", "745 tokens"),
        ("📤 Output Units", "10,000"),
        ("⚡ Task", "Next Word Prediction"),
        ("📦 Dataset", "qoute_dataset.csv"),
        ("🔤 Input", "Seed text"),
        ("📊 Params", "5.6M parameters"),
    ]

    for k, v in specs:
        color = "#10b981" if "10,000" in v or "8,725" in v else "#a78bfa"
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
            padding:0.6rem 0.85rem;margin-bottom:0.35rem;
            background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
            border-radius:12px;transition:all 0.2s'>
            <span style='font-size:0.75rem;color:#4b5563'>{k}</span>
            <span style='font-size:0.75rem;font-weight:600;color:{color}'>{v}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    temperature = st.slider("🌡️ Creativity Level", 0.3, 1.5, 0.8, 0.1,
        help="Low = predictable & coherent | High = creative & surprising")

    st.markdown(f"""
    <div style='background:rgba(109,40,217,0.08);border:1px solid rgba(109,40,217,0.18);
        border-radius:14px;padding:1rem;margin-top:1rem'>
        <div style='font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#6d28d9;margin-bottom:0.7rem'>
            🌡️ Temperature Guide
        </div>
        <div style='display:flex;flex-direction:column;gap:0.4rem'>
            <div style='display:flex;justify-content:space-between'>
                <span style='font-size:0.75rem;color:#4b5563'>0.3 - 0.5</span>
                <span style='font-size:0.75rem;color:#38bdf8'>Predictable</span>
            </div>
            <div style='display:flex;justify-content:space-between'>
                <span style='font-size:0.75rem;color:#4b5563'>0.6 - 0.9</span>
                <span style='font-size:0.75rem;color:#a78bfa'>Balanced ✨</span>
            </div>
            <div style='display:flex;justify-content:space-between'>
                <span style='font-size:0.75rem;color:#4b5563'>1.0 - 1.5</span>
                <span style='font-size:0.75rem;color:#f472b6'>Creative 🔥</span>
            </div>
        </div>
    </div>

    <div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
        border-radius:14px;padding:1rem;margin-top:0.8rem'>
        <div style='font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#374151;margin-bottom:0.7rem'>💡 Tips</div>
        <div style='color:#374151;font-size:0.78rem;line-height:2'>
        • Use 2-4 meaningful seed words<br>
        • Try different temperatures<br>
        • Quotes dataset = inspirational text<br>
        • More words = longer output
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='badge'>◆ &nbsp; Deep Learning &nbsp;·&nbsp; LSTM &nbsp;·&nbsp; NLP &nbsp;·&nbsp; RRN Part 3 &nbsp; ◆</div>
    <div class='hero-title'>Next Word<br><span class='gt'>Predictor</span></div>
    <div class='hero-sub'>AI-powered text generation · Trained on inspirational quotes · Bidirectional LSTM</div>
</div>
""", unsafe_allow_html=True)

# ── LOAD ───────────────────────────────────────────────────────────────────────
try:
    model, tokenizer, max_len, vocab_size = load_artifacts()
    model_ok = True
except Exception as e:
    st.error(f"❌ Error: {e}")
    model_ok = False

if model_ok:
    # STATS ROW
    st.markdown(f"""
    <div class='stats'>
        <div class='sc'><div class='sv'>{vocab_size:,}</div><div class='sk'>Vocab Size</div></div>
        <div class='sc'><div class='sv'>{max_len}</div><div class='sk'>Sequence Length</div></div>
        <div class='sc'><div class='sv'>128</div><div class='sk'>LSTM Units</div></div>
        <div class='sc'><div class='sv'>5.6M</div><div class='sk'>Parameters</div></div>
        <div class='sc'><div class='sv'>✅</div><div class='sk'>Model Ready</div></div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("<div class='pl'>Seed Text</div>", unsafe_allow_html=True)

        if 'sv' not in st.session_state:
            st.session_state['sv'] = ''

        seed = st.text_input("", value=st.session_state.get('sv',''),
            placeholder="e.g.  life is   |   the best way   |   success comes from",
            key="seed_input")

        num_words = st.slider("📝 Words to Generate", 5, 50, 15)

        c1, c2 = st.columns([3, 1])
        with c1:
            gen_btn = st.button("✨  Generate Text", use_container_width=True)
        with c2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='pl'>Quick Seeds</div>", unsafe_allow_html=True)

        q1, q2 = st.columns(2)
        q3, q4 = st.columns(2)
        with q1:
            if st.button("💭 life is", use_container_width=True):
                st.session_state['sv'] = "life is"
                st.rerun()
        with q2:
            if st.button("🌟 success comes", use_container_width=True):
                st.session_state['sv'] = "success comes from"
                st.rerun()
        with q3:
            if st.button("❤️ love is the", use_container_width=True):
                st.session_state['sv'] = "love is the"
                st.rerun()
        with q4:
            if st.button("🧠 the human mind", use_container_width=True):
                st.session_state['sv'] = "the human mind"
                st.rerun()

    with right:
        st.markdown("<div class='pl'>Generated Output</div>", unsafe_allow_html=True)

        if gen_btn:
            if not seed.strip():
                st.warning("⚠️ Please enter some seed text first!")
            else:
                with st.spinner("✨ Generating with LSTM..."):
                    time.sleep(0.4)
                    output = generate_text(seed.strip(), num_words, model, tokenizer, max_len, temperature)

                rest = output[len(seed.strip()):]
                words = rest.strip().split()

                st.markdown(f"""
                <div class='output-box'>
                    <div class='out-label'>✦ Generated Text</div>
                    <div class='out-text'>
                        <span class='seed-hl'>{seed.strip()}</span>{rest}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Word chips
                st.markdown("<div class='pl'>Generated Words</div>", unsafe_allow_html=True)
                chips_html = "".join([
                    f"<span class='chip' style='animation-delay:{i*0.05}s'>{w}</span>"
                    for i, w in enumerate(words)
                ])
                st.markdown(f"<div class='chips'>{chips_html}</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.text_area("📋 Copy Output", output, height=80, key="copy_out")

                with st.expander("📊 Generation Details"):
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Seed Words", len(seed.strip().split()))
                    d2.metric("Generated", num_words)
                    d3.metric("Temperature", temperature)
                    d4.metric("Total Words", len(output.split()))

        else:
            st.markdown("""
            <div class='empty'>
                <div class='empty-icon'>✨</div>
                <div class='empty-txt'>Enter a seed text on the left<br>and click Generate Text</div>
            </div>
            """, unsafe_allow_html=True)

        if clear_btn:
            st.session_state['sv'] = ''
            st.rerun()

st.markdown("""
<div style='text-align:center;padding:3rem 1rem 1.5rem;color:#0d0d1a;font-size:0.72rem;
    letter-spacing:2px;position:relative;z-index:1'>
    BUILT WITH ❤️ &nbsp;·&nbsp; NEXT WORD PREDICTOR &nbsp;·&nbsp; LSTM &nbsp;·&nbsp; RRN PART 3 OF DL
</div>
""", unsafe_allow_html=True)