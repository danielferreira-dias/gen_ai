from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
docs = [
    "./porto.md",
]

# ChromaDB configuration
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "porto_docs"

print(f"Using ChromaDB with collection: {COLLECTION_NAME}")
class DocumentTokenizer:
    def __init__(self, model_id: str, max_tokens: int):
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(model_id),
            max_tokens=max_tokens
        )
class DocumentChunker:
    def __init__(self, tokenizer: DocumentTokenizer):
        self.chunker = HybridChunker(
            tokenizer=tokenizer.tokenizer,
            merge_peers=True
        )
    
    def chunk_doc(self, document) -> list:
        chunk_iter = self.chunker.chunk(dl_doc=document)
        return list(chunk_iter)
    
    def contextualize(self, chunk):
        return self.chunker.contextualize(chunk=chunk)

def proccess_documents():
    # Initialize components
    tokenizer = DocumentTokenizer(model_id=EMBED_MODEL_ID, max_tokens=512)
    chunker = DocumentChunker(tokenizer=tokenizer)
    converter = DocumentConverter()

    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # FIXED: Create list to store LangChain Document objects
    langchain_documents = []
    
    for doc_path in docs:
        print(f"\nProcessing: {doc_path}")
        
        # Convert document
        result = converter.convert(doc_path)
        doc = result.document
        
        # Chunk document
        chunks = chunker.chunk_doc(doc)
        print(f"Generated {len(chunks)} chunks")
        
        # Embed each chunk
        for i, chunk in enumerate(chunks):
            enriched_text = chunker.contextualize(chunk=chunk)

            langchain_doc = Document(
                page_content=enriched_text,
                metadata={
                    "source": doc_path,
                    "chunk_index": i,
                    "chunk_id": f"{doc_path}_{i}",
                }
            )
            
            langchain_documents.append(langchain_doc)

            if i == 0:  #
                print(f"First chunk preview: {enriched_text}...")
    
    print(f"\n✓ Total chunks to store: {len(langchain_documents)}")

    # Create or connect to ChromaDB
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    print("Storing documents in ChromaDB...")
    vectorstore.add_documents(langchain_documents)
    print("✓ All chunks stored in ChromaDB!")

    return vectorstore

def query_vectorstore():
    """Query existing vectorstore"""
    print("\nConnecting to vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    print("✓ Connected to ChromaDB!")
    
    # Interactive query loop
    print("\n--- Vector Search Interface ---")
    print("Enter your query (or 'quit' to exit):")
    
    while True:
        user_query = input("\nQuery: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        try:
            results_with_scores = vectorstore.similarity_search_with_score(user_query, k=3)
            
            if not results_with_scores:
                print("No results found.")
                continue
            
            for i, (result, score) in enumerate(results_with_scores, 1):
                print(f"\n{'='*60}")
                print(f"Result {i} (Score: {score:.4f}):")
                print(f"Content: {result.page_content}...")
                print(f"Source: {result.metadata.get('source', 'Unknown')}")
                print(f"{'='*60}")
        except Exception as e:
            print(f"Error during search: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        # Run in query mode: python main.py query
        query_vectorstore()
    else:
        # Run in ingestion mode: python main.py
        proccess_documents()
        print("\n" + "="*60)
        print("Ingestion complete! Run 'python main.py query' to search.")
        print("="*60)
