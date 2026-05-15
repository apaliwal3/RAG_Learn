from src.data_loader import load_all_documents
from src.embeddings import EmbeddingPipeline

##Example

if __name__ == "__main__":
  docs = load_all_documents("data")
  embedding_pipeline = EmbeddingPipeline()
  chunks = embedding_pipeline.chunk_documents(docs)
  chunkvectors = embedding_pipeline.embed_chunks(chunks)
  print(chunkvectors)
