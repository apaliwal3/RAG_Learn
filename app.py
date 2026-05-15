from src.data_loader import load_all_documents
from src.embeddings import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

##Example

if __name__ == "__main__":
  #docs = load_all_documents("data")
  store = FaissVectorStore("faiss_store")
  #store.build_from_documents(docs)
  store.load()
  #print(store.query("Machine Learning", top_k=3))
  rag_search = RAGSearch()
  query = "What are the key concepts in Machine Learning?"
  response = rag_search.search(query, top_k=3)
  print(f"Summary:\n{response}")
