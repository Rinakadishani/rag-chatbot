"""
Answer generation module for RAG chatbot
"""

from typing import List, Dict, Any
from anthropic import Anthropic
import os


class AnswerGenerator:
    """Generates answers using Claude API with retrieved context"""
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        print("✓ Answer generator initialized")
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['metadata']['source']
            content = chunk['content']
            
            context_parts.append(f"[Document {i}: {source}]\n{content}")
            
            if source not in sources:
                sources.append(source)
        
        context_text = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on provided documents.

CONTEXT FROM DOCUMENTS:
{context_text}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question using ONLY information from the provided documents
2. If the documents don't contain enough information to answer, say so
3. Cite which document(s) you're using (e.g., "According to Document 1...")
4. Be concise but thorough
5. If the question is not related to the documents, politely decline to answer

Answer:"""
        
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.content[0].text
            
            return {
                'answer': answer,
                'sources': sources,
                'num_chunks_used': len(context_chunks),
                'model': self.model
            }
        
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'num_chunks_used': 0,
                'model': self.model,
                'error': str(e)
            }
    
    def check_relevance(self, query: str) -> bool:
        domain_keywords = [
            'health', 'medical', 'patient', 'doctor', 'hospital',
            'insurance', 'coverage', 'claim', 'policy', 'premium',
            'pharmaceutical', 'drug', 'medicine', 'clinical', 'trial',
            'treatment', 'diagnosis', 'therapy', 'care', 'prescription'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in domain_keywords)