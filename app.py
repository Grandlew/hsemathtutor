import os
from pathlib import Path
import gradio as gr

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LIOpenAI

NOTES_DIR = Path("./lecture_notes")
STORAGE_DIR = Path("./storage")

INDEX = None
ENGINE = None


def boot():
    global INDEX, ENGINE

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return "❌ OPENAI_API_KEY is not set (add it in Railway Variables)."

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Fewer chunks => faster indexing
    Settings.node_parser = SentenceSplitter(chunk_size=2400, chunk_overlap=50)

    # ✅ No torch needed
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    Settings.llm = LIOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.2,
    )

    # Load cached index if exists
    if any(STORAGE_DIR.iterdir()):
        sc = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        INDEX = load_index_from_storage(sc)
    else:
        pdfs = sorted(NOTES_DIR.glob("*.pdf"))
        if not pdfs:
            return "❌ No PDFs found in ./lecture_notes/ (upload PDFs and redeploy)."

        reader = SimpleDirectoryReader(
            input_dir=str(NOTES_DIR),
            required_exts=[".pdf"],
            file_extractor={".pdf": PDFReader()},
        )
        docs = reader.load_data()
        INDEX = VectorStoreIndex.from_documents(docs, show_progress=True)
        INDEX.storage_context.persist(persist_dir=str(STORAGE_DIR))

    ENGINE = INDEX.as_query_engine(similarity_top_k=4, response_mode="compact")
    return "✅ Online"


def tutor_answer(question: str, mode: str, level: str, show_sources: bool):
    global ENGINE
    if ENGINE is None:
        status = boot()
        if ENGINE is None:
            return f"🤖 {status}"

    retrieved = ENGINE.query(question)
    context = getattr(retrieved, "response", str(retrieved))

    system = f"""
You are an HSE Discrete Mathematics tutor.
Student level: {level}
Mode: {mode}

Rules:
- Be correct and clear.
- If ambiguous: ask ONE clarifying question first.
- Explain: definition + intuition + example + common mistake.
- Hint: do NOT give full solution; give 1–3 hints + checkpoint.
- Step-by-step: numbered steps with brief reasons.
- Quiz: ask 2–4 short questions and stop.
"""

    prompt = f"""{system}

Use the retrieved context below (may be partial):
---
{context}
---

Student question: {question}
Answer now:
"""

    final = ENGINE.query(prompt)
    answer = getattr(final, "response", str(final))

    sources_block = ""
    if show_sources:
        src_nodes = getattr(retrieved, "source_nodes", None)
        if src_nodes:
            items = []
            for sn in src_nodes[:4]:
                md = getattr(sn, "metadata", {}) or {}
                page = md.get("page_label") or md.get("page")
                fname = md.get("file_name") or md.get("filename") or "PDF"
                items.append(
                    f"- {fname}" + (f" (page {page})" if page is not None else ""))
            sources_block = "\n\n**Sources:**\n" + "\n".join(items)

    return answer + sources_block


def chat_fn(message, history, mode, level, show_sources):
    history = history or []
    if not (message or "").strip():
        return history, ""

    ans = tutor_answer(message, mode, level, show_sources)
    history.append((message, ans))
    return history, ""


with gr.Blocks(title="HSE Discrete Math Tutor") as demo:
    gr.Markdown("# 🧮 HSE Discrete Math Tutor")

    with gr.Row():
        mode = gr.Dropdown(["Explain", "Hint", "Step-by-step",
                           "Quiz"], value="Explain", label="Tutor mode")
        level = gr.Dropdown(["Beginner", "HSE midterm level", "Exam level"],
                            value="HSE midterm level", label="Difficulty")
        show_sources = gr.Checkbox(True, label="Show sources")

    chatbot = gr.Chatbot(height=520)
    msg = gr.Textbox(
        placeholder="Ask about logic, sets, graphs, combinatorics…")
    send = gr.Button("Send")

    msg.submit(chat_fn, inputs=[msg, chatbot, mode,
               level, show_sources], outputs=[chatbot, msg])
    send.click(chat_fn, inputs=[msg, chatbot, mode,
               level, show_sources], outputs=[chatbot, msg])

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
