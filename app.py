import streamlit as st
from rag import RAGChatbot
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
  --bg-main: #0f1115;
  --bg-panel: #161a22;
  --bg-card: #1c2230;
  --accent: #5b8cff;
  --accent-soft: rgba(91,140,255,.15);
  --text-main: #e8ebf0;
  --text-muted: #9aa4b2;
  --border: #2a3142;
}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp, .main { background: var(--bg-main); color: var(--text-main); }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#141826,#0f1115); }
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* Header */
.app-header {
  background: linear-gradient(135deg,#1b2140,#101426);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2rem;
  margin-bottom: 2rem;
}
.app-title { font-size: 2.6rem; font-weight: 800; }
.app-sub { color: var(--text-muted); margin-top: .4rem; }

/* Chat bubbles */
.chat-wrap { max-width: 980px; margin: 0 auto; }
.msg { padding: 1.1rem 1.3rem; border-radius: 16px; margin: .8rem 0; }
.msg-user { background: linear-gradient(135deg,#5b8cff,#3f6ee8); margin-left: 18%; }
.msg-bot { background: var(--bg-card); border: 1px solid var(--border); margin-right: 18%; }
.msg-label { font-size: .8rem; opacity: .8; margin-bottom: .3rem; }

/* Sources */
.sources {
  background: var(--accent-soft);
  border-left: 4px solid var(--accent);
  border-radius: 12px;
  padding: .8rem 1rem;
  margin-top: .6rem;
  font-size: .9rem;
}

/* Input */
.input-wrap {
  position: sticky;
  bottom: 0;
  background: linear-gradient(180deg,transparent,var(--bg-main) 35%);
  padding-top: 1.2rem;
}
.stTextInput input {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: .9rem 1rem !important;
  color: var(--text-main) !important;
}
.stButton button {
  background: linear-gradient(135deg,#5b8cff,#3f6ee8);
  border-radius: 14px;
  font-weight: 700;
}

/* Cards */
.card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 16px; padding: 1rem; margin-bottom: 1rem; }
.card h4 { margin-bottom: .6rem; }

/* Hide branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if 'chatbot' not in st.session_state:
    with st.spinner("Loading..."):
        st.session_state.chatbot = RAGChatbot()

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Filters")
    st.markdown("<small style='color:#9aa4b2;'>Select document categories to search:</small>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    healthcare = st.checkbox("Healthcare", True, key="filter_healthcare")
    insurance = st.checkbox("Insurance", True, key="filter_insurance")
    pharma = st.checkbox("Pharmaceutical", True, key="filter_pharma")
    
    active_filters = []
    if healthcare:
        active_filters.append("healthcare")
    if insurance:
        active_filters.append("insurance")
    if pharma:
        active_filters.append("pharmaceutical")
    
    if len(active_filters) < 3:
        st.markdown(f"<small style='color:#5b8cff;'>✓ {len(active_filters)} categories selected</small>", unsafe_allow_html=True)
    else:
        st.markdown(f"<small style='color:#5b8cff;'>✓ All categories</small>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### About")
    st.markdown("• 30 curated documents  \n• 1,400+ chunks indexed  \n• Finance, Healthcare, Supply Chain")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div class="app-title">RAG Chatbot</div>
  <div class="app-sub">Ask questions across financial, healthcare & supply‑chain documents — answers with sources.</div>
</div>
""", unsafe_allow_html=True)

chat = st.container()
with chat:
    st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown("<p style='color:#9aa4b2;text-align:center;'>Start by asking a question</p>", unsafe_allow_html=True)
    for m in st.session_state.messages:
        if m['role'] == 'user':
            st.markdown(f"""
            <div class="msg msg-user">
              <div class="msg-label">You</div>
              {m['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg msg-bot">
              <div class="msg-label">Assistant</div>
              {m['content']}
            </div>
            """, unsafe_allow_html=True)
            if m.get('sources'):
                st.markdown("<div class='sources'><b>📚 Sources</b><br>" + "<br>".join([f"• {s}" for s in m['sources']]) + "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='input-wrap'>", unsafe_allow_html=True)
col1, col2 = st.columns([7,1])
with col1:
    user_input = st.text_input("Ask", placeholder="Ask a question about the documents…", label_visibility="collapsed")
with col2:
    send = st.button("Send ➜", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if send and user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    
    categories = []
    if st.session_state.get('filter_healthcare', True):
        categories.append("healthcare")
    if st.session_state.get('filter_insurance', True):
        categories.append("insurance")
    if st.session_state.get('filter_pharma', True):
        categories.append("pharmaceutical")
    
    with st.spinner("Generating answer..."):
        result = st.session_state.chatbot.ask(
            user_input, 
            n_results=5,
            categories=categories if categories else None
        )
        st.session_state.messages.append({
            "role":"assistant",
            "content": result['answer'],
            "sources": result.get('sources', [])
        })
    st.rerun()