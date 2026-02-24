from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("./hf_cache").absolute()))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path("./hf_cache").absolute()))


NOTES_DIR = Path("./lecture_notes")
STORAGE_DIR = Path("./storage")


def main():
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(NOTES_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in ./lecture_notes/")

    # Speed settings (fewer chunks => fewer embeddings)
    Settings.node_parser = SentenceSplitter(chunk_size=2400, chunk_overlap=50)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embed_batch_size=128,
    )

    reader = SimpleDirectoryReader(
        input_dir=str(NOTES_DIR),
        required_exts=[".pdf"],
        file_extractor={".pdf": PDFReader()},
    )
    docs = reader.load_data()

    print(f"Loaded {len(docs)} document objects")
    total_chars = sum(len(getattr(d, "text", "")) for d in docs)
    print(f"Total extracted characters: {total_chars:,}")

    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print("✅ Index saved to ./storage")


if __name__ == "__main__":
    main()
