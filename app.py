import streamlit as st

st.set_page_config(page_title="GÉANT AI", page_icon="✨", layout="centered")

# --- Custom CSS (brand colors + rounded container + pill buttons) ---
st.markdown(
    """
    <style>
      :root{
        --geant: #7A1E4B;     /* deep magenta */
        --geant-2: #B54A7A;   /* lighter magenta */
        --bg: #ffffff;
        --soft: #F6F2F4;
        --text: #1f1f1f;
      }

      /* tighten top padding a bit */
      .block-container { padding-top: 1.2rem; }

      /* hide Streamlit default chrome */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* "card" container */
      .geant-card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 18px;
        padding: 18px 18px 10px 18px;
        background: var(--bg);
        box-shadow: 0 2px 18px rgba(0,0,0,0.05);
      }

      /* top bar */
      .geant-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 6px 14px 6px;
        border-bottom: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 16px;
      }

      .geant-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        letter-spacing: 0.3px;
        color: var(--geant);
        font-size: 22px;
      }

      .geant-icon {
        width: 34px; height: 34px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, rgba(122,30,75,0.12), rgba(181,74,122,0.12));
        color: var(--geant);
        font-size: 18px;
      }

      .geant-actions {
        display: flex;
        gap: 10px;
        color: var(--geant);
        font-size: 18px;
        user-select: none;
      }

      /* suggested prompt buttons styling */
      div.stButton > button {
        width: 100%;
        border-radius: 999px !important;
        border: 2px solid var(--geant) !important;
        color: var(--geant) !important;
        background: rgba(122,30,75,0.06) !important;
        padding: 0.7rem 1rem !important;
        font-weight: 600 !important;
      }
      div.stButton > button:hover {
        background: rgba(122,30,75,0.10) !important;
        border-color: var(--geant) !important;
      }

      /* chat input a bit rounder */
      section[data-testid="stChatInput"] textarea {
        border-radius: 999px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- UI wrapper card ---
st.markdown('<div class="geant-card">', unsafe_allow_html=True)

# Top bar (logo/title + icons). Icons are visual only.
st.markdown(
    """
    <div class="geant-topbar">
      <div class="geant-actions" title="Refresh">↻</div>
      <div class="geant-title">
        <div class="geant-icon">✨</div>
        <div>GÉANT AI</div>
      </div>
      <div class="geant-actions" title="Close">✕</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("**Hello! 👋 I am the GÉANT chatbot, your AI helper.**")
st.markdown("How can I help you today?")

# Suggested prompts
suggestions = [
    "Please summarize the SURF case study for me",
    "Show me the Annual Report of 2020",
    "How many universities does GÉANT collaborate with?",
]

cols = st.columns(1)
for s in suggestions:
    if st.button(s):
        st.session_state["pending_prompt"] = s

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# If user clicked a suggestion, push it into input flow
pending = st.session_state.pop("pending_prompt", None) if "pending_prompt" in st.session_state else None

prompt = st.chat_input("Ask anything ...")
if pending and not prompt:
    prompt = pending

if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder assistant response (replace later with your RAG)
    answer = "Got it — once the RAG backend is connected, I’ll answer with sources."
    st.session_state.messages.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

st.markdown("</div>", unsafe_allow_html=True)
