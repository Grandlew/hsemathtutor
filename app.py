from llama_index.llms.openai import OpenAI as LIOpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
import os
from pathlib import Path
import gradio as gr

# Caches (important on hosts)
os.environ.setdefault("HF_HOME", str(Path("./hf_cache").absolute()))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path("./hf_cache").absolute()))


NOTES_DIR = Path("./lecture_notes")
STORAGE_DIR = Path("./storage")

# ---------- Index / Engine ----------


def ensure_index():
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Chunking tuned to reduce chunks (faster indexing)
    Settings.node_parser = SentenceSplitter(chunk_size=2400, chunk_overlap=50)

    # Local embedding model (CPU OK)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embed_batch_size=128,
    )

    # LLM = OpenAI (fast, ChatGPT-like)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "❌ OPENAI_API_KEY not set. Add it in your host Secrets/Variables."

    Settings.llm = LIOpenAI(model="gpt-4o-mini",
                            api_key=api_key, temperature=0.2)

    # Load existing index if present
    if any(STORAGE_DIR.iterdir()):
        sc = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        idx = load_index_from_storage(sc)
        return idx, "✅ Loaded cached index"
    else:
        pdfs = sorted(NOTES_DIR.glob("*.pdf"))
        if not pdfs:
            return None, "❌ No PDFs found in ./lecture_notes/ (upload PDFs and restart)."

        reader = SimpleDirectoryReader(
            input_dir=str(NOTES_DIR),
            required_exts=[".pdf"],
            file_extractor={".pdf": PDFReader()},
        )
        docs = reader.load_data()
        idx = VectorStoreIndex.from_documents(docs, show_progress=True)
        idx.storage_context.persist(persist_dir=str(STORAGE_DIR))
        return idx, "✅ Built index (first time) and cached to ./storage"


def build_query_engine(idx):
    return idx.as_query_engine(similarity_top_k=4, response_mode="compact")


# Lazy globals
INDEX = None
ENGINE = None
BOOT_MSG = "Starting…"


def boot():
    global INDEX, ENGINE, BOOT_MSG
    if ENGINE is not None:
        return BOOT_MSG

    idx, msg = ensure_index()
    BOOT_MSG = msg
    if idx is None:
        return BOOT_MSG

    INDEX = idx
    ENGINE = build_query_engine(INDEX)
    BOOT_MSG = "✅ Online"
    return BOOT_MSG

# ---------- Tutor prompt wrapper ----------


def tutor_answer(question: str, mode: str, level: str, show_sources: bool):
    if ENGINE is None:
        msg = boot()
        return f"🤖 {msg}"

    # Retrieval
    resp = ENGINE.query(question)
    context = getattr(resp, "response", str(resp))

    # Tutor behavior
    system = f"""
You are an HSE Discrete Mathematics tutor.
Student level: {level}
Mode: {mode}

Rules:
- Be correct and clear.
- If ambiguous: ask ONE short clarifying question first.
- Explain mode: definition + intuition + small example + common mistake.
- Hint mode: do NOT give full solution; give 1–3 hints + a checkpoint.
- Step-by-step: numbered steps with brief reasons.
- Quiz: ask 2–4 short questions and stop.
"""

    # We ask the LLM to answer using the retrieved context.
    # LlamaIndex OpenAI integration uses Settings.llm behind the scenes when summarizing,
    # but we still guide style through the query prompt by embedding tutor instructions.
    prompt = f"""{system}

Use this retrieved notes context (may be partial):
---
{context}
---

Student question: {question}
Answer now:
"""

    # Ask again via engine so it uses LLM; simplest: query_engine.query(prompt)
    final = ENGINE.query(prompt)
    answer = getattr(final, "response", str(final))

    # Sources
    sources_block = ""
    if show_sources:
        src_nodes = getattr(resp, "source_nodes", None)
        if src_nodes:
            items = []
            for sn in src_nodes[:4]:
                md = getattr(sn, "metadata", {}) or {}
                page = md.get("page_label") or md.get("page")
                fname = md.get("file_name") or md.get("filename") or "PDF"
                if page is not None:
                    items.append(f"- {fname} (page {page})")
                else:
                    items.append(f"- {fname}")
            if items:
                sources_block = "\n\n**Sources:**\n" + "\n".join(items)

    return answer + sources_block

# ---------- Gradio ----------


def chat_fn(message, history, mode, level, show_sources):
    history = history or []
    if not (message or "").strip():
        return history, "", boot_status.value

    ans = tutor_answer(message, mode, level, show_sources)
    history.append((message, ans))
    return history, "", boot()


def rebuild_index():
    # Danger: can be slow on host. Best done locally using build_index.py.
    global INDEX, ENGINE, BOOT_MSG
    BOOT_MSG = "🧠 Rebuilding index…"
    INDEX = None
    ENGINE = None

    # wipe storage
    if STORAGE_DIR.exists():
        for p in STORAGE_DIR.glob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except:
                pass

    msg = boot()
    return msg


with gr.Blocks(title="HSE Discrete Math Tutor") as demo:
    gr.Markdown(
        "# 🧮 HSE Discrete Math Tutor (ChatGPT-like)\nRAG over lecture notes + problem sets/solutions.")

    boot_status = gr.Markdown("Starting…")
    demo.load(fn=boot, inputs=None, outputs=boot_status)

    with gr.Row():
        mode = gr.Dropdown(["Explain", "Hint", "Step-by-step",
                           "Quiz"], value="Explain", label="Tutor mode")
        level = gr.Dropdown(["Beginner", "HSE midterm level", "Exam level"],
                            value="HSE midterm level", label="Difficulty")
        show_sources = gr.Checkbox(True, label="Show sources")

    chatbot = gr.Chatbot(height=520)
    msg = gr.Textbox(
        placeholder="Ask about logic, sets, graphs, combinatorics…", scale=8)
    send = gr.Button("Send", scale=1)

    with gr.Accordion("Admin", open=False):
        gr.Markdown(
            "If you changed PDFs, rebuild the index (slow on host). Prefer running build_index.py locally.")
        rebuild_btn = gr.Button("Rebuild index now (slow)")
        rebuild_btn.click(fn=rebuild_index, inputs=None, outputs=boot_status)

    msg.submit(chat_fn, inputs=[msg, chatbot, mode, level, show_sources], outputs=[
               chatbot, msg, boot_status])
    send.click(chat_fn, inputs=[msg, chatbot, mode, level, show_sources], outputs=[
               chatbot, msg, boot_status])

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
