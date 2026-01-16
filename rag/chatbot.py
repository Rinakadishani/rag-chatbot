from typing import Dict, Any
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .retriever import HybridRetriever
from .generator import AnswerGenerator
from .document_loader import DocumentLoader
from .chunker import TextChunker


class RAGChatbot:
    
    def __init__(self, api_key: str = None):
        print("Initializing RAG Chatbot...")
        print("-" * 60)
        
        print("Loading embedding model...")
        self.embedding_model = EmbeddingModel()
        
        print("Loading vector database...")
        self.vector_store = VectorStore()
        
        print("Loading documents for keyword search...")
        loader = DocumentLoader()
        docs = loader.load_directory("./documents")
        chunker = TextChunker()
        self.chunks = chunker.chunk_documents(docs)
        
        print("Initializing hybrid retriever...")
        self.retriever = HybridRetriever(
            self.vector_store,
            self.embedding_model,
            self.chunks
        )
        
        print("Initializing answer generator...")
        self.generator = AnswerGenerator(api_key=api_key)
        
        print("-" * 60)
        print("RAG Chatbot ready!")
        print(f"Vector database: {self.vector_store.get_count()} chunks")
        print("-" * 60)
    
    def ask(
        self,
        question: str,
        n_results: int = 5,
        verbose: bool = False
    ) -> Dict[str, Any]:

        if verbose:
            print(f"\nQuestion: {question}")
            print("-" * 60)
    
        if not self.generator.check_relevance(question):
            return {
                'answer': "I'm sorry, but your question doesn't seem to be related to healthcare, insurance, or pharmaceutical topics. I can only answer questions about these domains based on the documents I have access to.",
                'sources': [],
                'retrieved_chunks': 0,
                'relevant': False
            }
        
        # Retrieve relevant chunks
        if verbose:
            print(f"Retrieving top {n_results} relevant chunks...")
        
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            n_results=n_results
        )
        
        if verbose:
            print(f"Retrieved {len(retrieved_chunks)} chunks")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"\n  Chunk {i} (score: {chunk['score']:.3f})")
                print(f"  Source: {chunk['metadata']['source']}")
                print(f"  Preview: {chunk['content'][:100]}...")
        
        if verbose:
            print("\nGenerating answer with Claude API...")
        
        result = self.generator.generate_answer(
            query=question,
            context_chunks=retrieved_chunks
        )
        
        result['retrieved_chunks'] = len(retrieved_chunks)
        result['relevant'] = True
        
        if verbose:
            print("Answer generated")
            print("-" * 60)
        
        return result