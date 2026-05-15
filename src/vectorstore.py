import os
import faiss
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embeddings import EmbeddingPipeline

class FaissVectorStore:
  def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):
    self.persist_dir = persist_dir
    os.makedirs(self.persist_dir, exist_ok=True)
    self.index = None
    self.metadata = []
    self.embedding_model = embedding_model
    self.model = SentenceTransformer(embedding_model)
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    print(f"Loaded embedding model: {embedding_model}")
  
  def build_from_documents(self, documents: List[Any]):
    print(f"Building Faiss index from {len(documents)} documents")
    embedding_pipeline = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
    chunks = embedding_pipeline.chunk_documents(documents)
    embeddings = embedding_pipeline.embed_chunks(chunks)
    metadatas = [{"text": chunk.page_content for chunk in chunks}]
    self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
    self.save()
    print(f"Built Faiss index with {len(chunks)} chunks and saved to {self.persist_dir}")
  
  def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
    dim = embeddings.shape[1]
    if self.index is None:
      self.index = faiss.IndexFlatL2(dim)
      print(f"Created new Faiss index with dimension: {dim}")
    self.index.add(embeddings)
    if metadatas:
      self.metadata.extend(metadatas)
    print(f"Added {embeddings.shape[0]} embeddings to Faiss index")
  
  def