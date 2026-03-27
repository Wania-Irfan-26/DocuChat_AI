# 💬 DocuChat AI

A Streamlit-based RAG (Retrieval-Augmented Generation) app that lets you upload documents and have an intelligent conversation with them — powered by LangChain, Groq LLaMA 3.1, and HuggingFace embeddings.

---

## Features

- Upload PDF, TXT, or DOCX files
- Automatic document chunking and embedding
- Semantic search via Chroma vector store
- Chat interface with user/AI message bubbles
- Powered by Groq's LLaMA 3.1 8B model
- Clean white UI with Poppins font

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq — LLaMA 3.1 8B Instant |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | Chroma (in-memory) |
| RAG Framework | LangChain |
| File Support | PDF, TXT, DOCX |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/docuchat-ai.git
cd docuchat-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com).

### 4. Run the app

```bash
streamlit run app.py
```

---

## Usage

1. Upload one or more PDF, TXT, or DOCX files from the sidebar
2. Click **Process Documents** to index them
3. Type your question in the chat input and press **Enter** or click **Send**
4. Use **Reset Chat** to clear the conversation (documents stay indexed)

---

## Project Structure

```
docuchat-ai/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
└── README.md
```

---

## Notes

- The `.env` file should never be committed to version control. Add it to `.gitignore`.
- The vector store is in-memory and resets on each session.
- Larger documents may take a moment to embed on first load.
