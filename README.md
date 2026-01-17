cat > README.md << 'EOF'

# RAG Healthcare Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions about healthcare, insurance, and pharmaceuticals.

---

## Features

### Core Capabilities

- **Intelligent Q&A**: Answers questions using verified document sources
- **Source Citations**: Always cites which documents were used
- **Multi-Domain**: Covers healthcare, insurance, and pharmaceutical topics
- **Hybrid Search**: Combines semantic (vector) and keyword (BM25) search
- **Metadata Filtering**: Filter results by document category
- **Guardrails**: Refuses off-topic questions outside the domain

### Technical Features

- **Vector Database**: ChromaDB with 1,400+ indexed chunks
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Claude Sonnet 4 (Anthropic)
- **UI**: Modern dark-theme Streamlit interface
- **Architecture**: Clean, modular Python codebase

---

## Tech Stack

| Component      | Technology                  |
| -------------- | --------------------------- |
| **LLM**        | Claude Sonnet 4 (Anthropic) |
| **Vector DB**  | ChromaDB                    |
| **Embeddings** | sentence-transformers       |
| **Search**     | Hybrid (Semantic + BM25)    |
| **Frontend**   | Streamlit                   |
| **Language**   | Python 3.9+                 |

---

## Project Structure

```
rag-chatbot/
├── rag/                          # Core RAG modules
│   ├── __init__.py              # Package initialization
│   ├── document_loader.py       # PDF/document loading with category tagging
│   ├── chunker.py               # Text chunking with overlap
│   ├── embeddings.py            # Embedding generation
│   ├── vector_store.py          # ChromaDB interface with filtering
│   ├── retriever.py             # Hybrid search (semantic + BM25)
│   ├── generator.py             # Answer generation with Claude
│   └── chatbot.py               # Main orchestrator
├── documents/                    # Knowledge base (30 PDFs)
├── chroma_db/                   # Vector database (gitignored)
├── app.py                       # Streamlit web interface
├── ingest_documents.py          # Document ingestion script
├── requirements.txt             # Python dependencies
├── .env                         # API keys (gitignored)
├── .env.example                 # Environment template
└── README.md                    # This file
```

---

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Rinakadishani/rag-chatbot.git
cd rag-chatbot
```

**2. Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure API key**

```bash
# Copy template
cp .env.example .env

# Edit .env and add your API key
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

**5. Ingest documents**

```bash
python3 ingest_documents.py
```

_This takes 5-10 minutes and creates the vector database._

**6. Run the application**

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## Usage

### Basic Questions

```
"What are pharmaceutical services in primary healthcare?"
"How does health insurance coverage work?"
"What are the phases of clinical trials?"
```

### Using Filters

1. Open sidebar
2. Check/uncheck categories:
   - Healthcare
   - Insurance
   - Pharmaceutical
3. Ask your question
4. Results will only include selected categories

### Off-Topic Questions

The chatbot will refuse questions unrelated to its domain:

```
User: "What's the weather today?"
Bot: "I can only answer questions about healthcare, insurance, or pharmaceuticals."
```

---

## System Architecture

```
┌─────────────┐
│   User      │
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   Streamlit UI              │
│   (app.py)                  │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   RAG Chatbot               │
│   (chatbot.py)              │
└──────┬──────────────────────┘
       │
       ├──► [Check Relevance]
       │    (generator.py)
       │
       ├──► [Hybrid Retrieval]
       │    ├─ Semantic Search (ChromaDB)
       │    └─ Keyword Search (BM25)
       │    (retriever.py)
       │
       └──► [Answer Generation]
            (Claude API)
            (generator.py)
```

---

### Metadata Filtering

Documents are tagged during ingestion:

- **Healthcare**: health, medical, patient, care
- **Insurance**: insurance, coverage, policy, claim
- **Pharmaceutical**: pharma, drug, trial, medicine

Users can filter searches by these categories.

---

## Performance

- **Response Time**: 3-5 seconds per query
- **Database Size**: ~500MB for 1,400 chunks
- **Concurrent Users**: Supports multiple users (Streamlit sessions)
