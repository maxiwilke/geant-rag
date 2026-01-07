import time
import uuid
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="GÉANT AI", page_icon="✨", layout="centered")

# -----------------------------
# Theme / CSS (approx. Figma)
# -----------------------------
PRIMARY = "#810947"
USER_BG = "#ffd9e9"
BOT_TEXT = "#464646"

st.markdown(
    f"""
    <style>
      .stApp {{ background: #ffffff; }} /* ensure overall page is white */
      .block-container {{ padding-top: 1.0rem; max-width: 720px; }}
      #MainMenu, footer, header {{ visibility: hidden; }}

      .geant-shell {{
        width: 600px;
        margin: 0 auto;
        background: white;
        border-radius: 18px;
        box-shadow: 0 14px 40px rgba(0,0,0,0.12);
        overflow: hidden;
        border: 1px solid rgba(70,70,70,0.25);
      }}

      /* Streamlit buttons styled as "pills" */
      div.stButton > button {{
        border-radius: 999px !important;
        border: 2px solid {PRIMARY} !important;
        background: {USER_BG} !important;
        color: {PRIMARY} !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 12px 18px !important;
        width: auto !important;
        max-width: 100% !important;
        text-align: left !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
      }}
      div.stButton > button:hover {{ opacity: 0.85; }}

      .geant-title {{
        font-weight: 800;
        color: {PRIMARY};
        letter-spacing: 0.3px;
        font-size: 22px;
        display: flex;
        gap: 10px;
        align-items: center;
      }}
      .geant-badge {{
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        background: rgba(129,9,71,0.10);
      }}

      /* Add separator line above input */
      .stChatInput {{
        border-top: 1px solid rgba(70,70,70,0.12) !important;
        padding-top: 16px !important;
      }}

      /* Fix chat input styling */
      section[data-testid="stChatInput"] {{
        background: transparent !important;
        border: none !important;
        padding-top: 16px !important;
      }}
      
      section[data-testid="stChatInput"] > div {{
        background: transparent !important;
      }}

      /* Input field container */
      section[data-testid="stChatInput"] > div > div {{
        background: transparent !important;
      }}

      section[data-testid="stChatInput"] textarea {{
        border-radius: 999px !important;
        border: 2px solid {PRIMARY} !important;
        background: white !important;
        font-size: 18px !important;
        padding: 14px 20px !important;
        transition: all 0.2s ease !important;
        color: #333 !important;
      }}

      section[data-testid="stChatInput"] textarea:focus {{
        border: 2px solid {PRIMARY} !important;
        box-shadow: none !important;
        outline: none !important;
      }}

      section[data-testid="stChatInput"] textarea::placeholder {{
        color: #999 !important;
      }}

      /* Style the send button */
      section[data-testid="stChatInput"] button[kind="primary"],
      section[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] {{
        border-radius: 50% !important;
        background: {PRIMARY} !important;
        border: none !important;
        width: 48px !important;
        height: 48px !important;
        min-width: 48px !important;
        min-height: 48px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin-left: 10px !important;
      }}

      section[data-testid="stChatInput"] button[kind="primary"]:hover,
      section[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:hover {{
        opacity: 0.85 !important;
        background: {PRIMARY} !important;
      }}

      section[data-testid="stChatInput"] button[kind="primary"] svg,
      section[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] svg {{
        width: 24px !important;
        height: 24px !important;
        color: white !important;
        fill: white !important;
      }}

      .center-screen {{
        min-height: 70vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 14px;
      }}
      .spinner {{
        width: 64px;
        height: 64px;
        border: 4px solid #E5E7EB;
        border-top-color: {PRIMARY};
        border-radius: 999px;
        animation: spin 1s linear infinite;
      }}
      @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
      }}
      .spinner-text {{ color: {PRIMARY}; font-size: 18px; font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Session state init
# -----------------------------
def init_state():
    if "is_open" not in st.session_state:
        st.session_state.is_open = True
    if "is_loading_link" not in st.session_state:
        st.session_state.is_loading_link = False
    if "is_thinking" not in st.session_state:
        st.session_state.is_thinking = False
    if "hide_recommendations" not in st.session_state:
        st.session_state.hide_recommendations = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.session_state.messages = [
            {"id": "1", "text": "Hello! 👋 I am the GÉANT chatbot, your AI helper.", "sender": "bot"},
            {"id": "2", "text": "How can I help you today?", "sender": "bot"},
        ]

init_state()

# -----------------------------
# Placeholder answers (no real LLM yet)
# -----------------------------
PLACEHOLDER_ANSWER = "Here would be an answer if I would be connected to an LLM."

def add_user_message(text: str):
    st.session_state.messages.append({"id": str(uuid.uuid4()), "text": text, "sender": "user"})

def add_bot_message(text: str):
    st.session_state.messages.append({"id": str(uuid.uuid4()), "text": text, "sender": "bot"})

def restart_chat():
    st.session_state.messages = [
        {"id": str(uuid.uuid4()), "text": "Hello! 👋 I am the GÉANT chatbot, your AI helper.", "sender": "bot"},
        {"id": str(uuid.uuid4()), "text": "How can I help you today?", "sender": "bot"},
    ]
    st.session_state.is_thinking = False
    st.session_state.hide_recommendations = False
    if "_recommendation_last" in st.session_state:
        del st.session_state["_recommendation_last"]

def close_chat():
    st.session_state.is_open = False

def reopen_chat():
    st.session_state.is_open = True

def send_logic(user_text: str):
    add_user_message(user_text)
    st.session_state.is_thinking = True
    st.session_state.hide_recommendations = True
    st.rerun()

# -----------------------------
# Screens: loading link / closed
# -----------------------------
if st.session_state.is_loading_link:
    st.markdown(
        """
        <div class="center-screen">
          <div class="spinner"></div>
          <div class="spinner-text">Redirect to source</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(5)
    st.session_state.is_loading_link = False
    st.rerun()

if not st.session_state.is_open:
    st.markdown(
        """
        <div class="center-screen">
          <div style="width: 650px; background:white; border-radius: 18px; box-shadow: 0 14px 40px rgba(0,0,0,0.12); padding: 26px; text-align:center;">
            <p style="color:#6b7280; font-size:18px; margin-bottom: 14px;">Chat closed</p>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Reopen Chat", key="reopen"):
        reopen_chat()
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# -----------------------------
# Main shell
# -----------------------------
st.markdown('<div class="geant-shell">', unsafe_allow_html=True)

# Header bar (use keys to avoid collisions)
h1, h2, h3 = st.columns([1, 6, 1])
with h1:
    if st.button("↻", help="Restart chat", key="restart"):
        restart_chat()
        st.rerun()
with h2:
    st.markdown(
        f"""
        <div class="geant-title" style="justify-content:center;">
          <div class="geant-badge">✨</div>
          <div>GÉANT AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with h3:
    if st.button("✕", help="Close chat", key="close"):
        close_chat()
        st.rerun()

def render_messages_html(messages, is_thinking: bool) -> str:
    def esc(s: str) -> str:
        return (
            (s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("\n", "<br/>")
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #ffffff;
                overflow: hidden;
            }}
            .messages-container {{
                padding: 16px;
                display: flex;
                flex-direction: column;
            }}
            .msg-row {{
                display: flex;
                margin-bottom: 10px;
            }}
            .msg-user {{ justify-content: flex-end; }}
            .msg-bot {{ justify-content: flex-start; }}
            .bubble {{
                max-width: 70%;
                padding: 10px 14px;
                border-radius: 18px;
                font-size: 18px;
                line-height: 1.35;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .bubble-user {{
                background: {USER_BG};
                color: {PRIMARY};
            }}
            .bubble-bot {{
                background: white;
                color: {BOT_TEXT};
                border: 1px solid rgba(70,70,70,0.12);
            }}
            @keyframes bounce {{
                0%, 60%, 100% {{ transform: translateY(0); }}
                30% {{ transform: translateY(-8px); }}
            }}
            .dot {{
                width: 10px;
                height: 10px;
                border-radius: 999px;
                background: {PRIMARY};
                display: inline-block;
            }}
            .dot1 {{ animation: bounce 1s infinite; animation-delay: 0.001s; }}
            .dot2 {{ animation: bounce 1s infinite; animation-delay: 0.18s; }}
            .dot3 {{ animation: bounce 1s infinite; animation-delay: 0.36s; }}
        </style>
    </head>
    <body>
        <div class="messages-container" id="msgContainer">
    """

    for m in messages:
        sender = m["sender"]
        cls_row = "msg-user" if sender == "user" else "msg-bot"
        cls_bubble = "bubble-user" if sender == "user" else "bubble-bot"
        html += f"""
            <div class="msg-row {cls_row}">
                <div class="bubble {cls_bubble}">{esc(m["text"])}</div>
            </div>
        """

    if is_thinking:
        html += f"""
            <div class="msg-row msg-bot" style="margin-top: 10px;">
                <div class="bubble bubble-bot" style="display:flex; gap:8px; align-items:center; padding: 14px;">
                    <span class="dot dot1"></span>
                    <span class="dot dot2"></span>
                    <span class="dot dot3"></span>
                </div>
            </div>
        """

    html += """
        </div>
        <script>
            // Send height to parent
            function updateHeight() {
                const container = document.getElementById('msgContainer');
                const height = container.offsetHeight;
                window.parent.postMessage({type: 'streamlit:setFrameHeight', height: height}, '*');
            }
            updateHeight();
            // Update again after images/content loads
            setTimeout(updateHeight, 100);
        </script>
    </body>
    </html>
    """
    return html

# Calculate approximate height based on message count
msg_count = len(st.session_state.messages)
if st.session_state.is_thinking:
    msg_count += 1
# Approximate: each message ~60px, plus padding
approx_height = max(100, min(500, (msg_count * 60) + 32))

# Render messages using iframe
components.html(
    render_messages_html(st.session_state.messages, st.session_state.is_thinking),
    height=approx_height,
    scrolling=False,
)

# Recommendation buttons (clickable) - Place inside the shell
if len(st.session_state.messages) <= 2 and not st.session_state.is_thinking and not st.session_state.hide_recommendations:
    st.markdown('<div style="padding: 0 16px 16px 16px;">', unsafe_allow_html=True)
    recs = [
        "Please summarize the SURF case study for me",
        "Show me the Annual Report of 2020",
        "How many universities does GEANT collaborate with?",
    ]
    for i, r in enumerate(recs):
        if st.button(r, key=f"rec_{i}"):
            add_user_message(r)
            st.session_state.is_thinking = True
            st.session_state.hide_recommendations = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# If thinking, simulate a short delay then add placeholder answer and stop thinking
if st.session_state.is_thinking:
    time.sleep(1.2)
    add_bot_message(PLACEHOLDER_ANSWER)
    st.session_state.is_thinking = False
    st.rerun()

# Close shell div
st.markdown("</div>", unsafe_allow_html=True)

# Input
prompt = st.chat_input("Ask anything ...")
if prompt:
    send_logic(prompt)