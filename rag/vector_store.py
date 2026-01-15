"""
Vector store module for RAG chatbot
Manages ChromaDB for storing and retrieving document chunks
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import uuid


class VectorStore:
    """Vector database for storing and retrieving document chunks"""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize vector store.
        """
        print(f"Initializing ChromaDB at {persist_directory}...")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Collection '{collection_name}' ready ({self.collection.count()} documents)")
    
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        print(f"Adding {len(chunks)} documents to vector store...")
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            if (i + batch_size) % 500 == 0:
                print(f"  Added {min(i + batch_size, len(chunks))}/{len(chunks)} documents...")
        
        print(f"Successfully added {len(chunks)} documents")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        formatted_results = {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
        }
        
        return formatted_results
    
    def get_count(self) -> int:
        return self.collection.count()
    
    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared")