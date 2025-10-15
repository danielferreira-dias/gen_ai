from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Get the path to the Porto markdown file
current_dir = os.path.dirname(os.path.abspath(__file__))
porto_file_path = os.path.join(current_dir, "porto.md")

# Load the Porto markdown document
loader = UnstructuredMarkdownLoader(porto_file_path)
docs_list = loader.load()

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

# Add the document chunks to the "vector store" using OpenAIEmbeddings
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
)

# With langchain we can easily turn any vector store into a retrieval component:
retriever = vectorstore.as_retriever(k=6)

# https://docs.langchain.com/langsmith/evaluate-chatbot-tutorial
# https://docs.langchain.com/langsmith/evaluate-rag-tutorial