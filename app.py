from src.data_loader import load_all_documents
from src.embeddings import EmbeddingPipeline
from src.vectorstore import FaissVectorStore

##Example

if __name__ == "__main__":
  docs = load_all_documents("data")
  store = FaissVectorStore("faiss_store")
  store.build_from_documents(docs)
  store.load()
  print(store.query("Machine Learning", top_k=3))
