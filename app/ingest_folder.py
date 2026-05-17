import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag import get_vector_store

# Path to knowledge base
DATA_PATH = "app/knowledge_base"


def load_text_files():
    """
    Load all text files from the knowledge base folder
    and convert them into LangChain Documents.
    """

    if not os.path.exists(DATA_PATH):
        raise ValueError(f"Knowledge base folder not found: {DATA_PATH}")

    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

                # simple category tagging from filename
                category = file.replace(".txt", "").lower()

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file,
                            "category": category,
                            "type": "medical_knowledge",
                        },
                    )
                )

    return docs


def main():
    """
    Build vector embeddings and upload them to Supabase.
    """

    print("Loading knowledge base...")

    raw_docs = load_text_files()

    print(f"Loaded {len(raw_docs)} documents")

    # Split large documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    split_docs = splitter.split_documents(raw_docs)

    print(f"Created {len(split_docs)} chunks")

    # Initialize vector store
    vector_store = get_vector_store()

    print("Uploading knowledge base to Supabase vector store...")

    vector_store.add_documents(split_docs)

    print("Upload complete!")


if __name__ == "__main__":
    main()