"""
Text chunking module for RAG chatbot
"""
from typing import List, Dict, Any
import tiktoken


class TextChunker:
    """Chunks text into smaller pieces with overlap"""
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
       
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            print("Warning: tiktoken encoding failed, using approximate token count")
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            #1 token = 4 characters
            return len(text) // 4
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        if self.encoding:
            tokens = self.encoding.encode(text)
            
            chunks = []
            start = 0
            chunk_id = 0
            
            while start < len(tokens):
                end = start + self.chunk_size
                chunk_tokens = tokens[start:end]
                
                chunk_text = self.encoding.decode(chunk_tokens)
                
                chunk = {
                    'content': chunk_text.strip(),
                    'metadata': {
                        **metadata,
                        'chunk_id': chunk_id,
                        'start_token': start,
                        'end_token': end,
                        'token_count': len(chunk_tokens)
                    }
                }
                chunks.append(chunk)
                
                start += self.chunk_size - self.chunk_overlap
                chunk_id += 1
        else:
            char_chunk_size = self.chunk_size * 4
            char_overlap = self.chunk_overlap * 4
            
            chunks = []
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = start + char_chunk_size
                chunk_text = text[start:end]
                
                chunk = {
                    'content': chunk_text.strip(),
                    'metadata': {
                        **metadata,
                        'chunk_id': chunk_id,
                        'char_count': len(chunk_text)
                    }
                }
                chunks.append(chunk)
                
                start += char_chunk_size - char_overlap
                chunk_id += 1
        
        return chunks
    
    def chunk_documents(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of chunks with metadata
        """
        all_chunks = []
        
        print("\nChunking documents...")
        for i, doc in enumerate(documents, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(documents)} documents...")
            chunks = self.chunk_text(doc.content, doc.metadata)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents\n")
        return all_chunks