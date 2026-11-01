import os
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class KnowledgeBaseRAG:
    def __init__(self, kb_dir="kb", persist_dir="./chroma_db"):
        self.kb_dir = kb_dir
        self.persist_dir = persist_dir
        # Using a fast, local HuggingFace embedding model (runs completely free on CPU)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self._build_or_load_index()

    def _build_or_load_index(self):
        # If Chroma index already exists, load it directly
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"Loading existing vector store from {self.persist_dir}...")
            self.vector_store = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            return

        print("Building vector index from kb/ markdown files...")
        documents = []
        for file in sorted(os.listdir(self.kb_dir)):
            if not file.endswith(".md"):
                continue
            
            # Rule: prioritize pol_returns_v4; ignore superseded versions
            if "superseded" in file:
                continue

            file_path = os.path.join(self.kb_dir, file)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Split document by markdown headings
            chunks = re.split(r'\n(?=###? |\n## )', content)
            for c in chunks:
                chunk_str = c.strip()
                if not chunk_str:
                    continue

                # Extract [SECTION-ID: ...] or bracketed heading IDs like [RET-3], [FAQ-1], [SEC-1]
                sec_match = re.search(r'\[(?:SECTION-ID:\s*)?([A-Za-z0-9_-]+)\]', chunk_str)
                sec_id = sec_match.group(1) if sec_match else file.replace(".md", "")

                documents.append({
                    "text": chunk_str,
                    "metadata": {
                        "source": file,
                        "section_id": sec_id
                    }
                })

        texts = [d["text"] for d in documents]
        metadatas = [d["metadata"] for d in documents]

        print(f"Indexing {len(texts)} chunks into ChromaDB...")
        self.vector_store = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print("Vector database built successfully.")

    def retrieve(self, query: str, top_k: int = 3):
        results = self.vector_store.similarity_search(query, k=top_k)
        retrieved_data = []
        for doc in results:
            retrieved_data.append({
                "chunk": doc.page_content,
                "section_id": doc.metadata.get("section_id", "UNKNOWN"),
                "source": doc.metadata.get("source", "")
            })
        return retrieved_data

if __name__ == "__main__":
    rag = KnowledgeBaseRAG()
    test_query = "What is the return policy for electronics?"
    chunks = rag.retrieve(test_query, top_k=2)
    print("\n--- Test Retrieval Output ---")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] Section: {c['section_id']} (Source: {c['source']})")
        print(f"Snippet: {c['chunk'][:140]}...\n")