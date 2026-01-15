"""
Document loader module for RAG chatbot
Supports PDF, DOCX, MD, TXT files
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2


class Document:
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self):
        return f"Document(source={self.metadata.get('source', 'unknown')}, length={len(self.content)})"


class DocumentLoader:
    """Loads documents from various file formats"""
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        documents = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        doc = Document(
                            content=text,
                            metadata={
                                'source': os.path.basename(file_path),
                                'page': page_num,
                                'file_type': 'pdf',
                                'file_path': file_path
                            }
                        )
                        documents.append(doc)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return documents
    
    @staticmethod
    def load_txt(file_path: str) -> List[Document]:
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
                if text.strip():
                    document = Document(
                        content=text,
                        metadata={
                            'source': os.path.basename(file_path),
                            'file_type': 'txt',
                            'file_path': file_path
                        }
                    )
                    documents.append(document)
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
        return documents
    
    @staticmethod
    def load_directory(directory_path: str) -> List[Document]:
        documents = []
        path = Path(directory_path)
        
        if not path.exists():
            print(f"Directory {directory_path} does not exist")
            return documents
        
        loaders = {
            '.pdf': DocumentLoader.load_pdf,
            '.txt': DocumentLoader.load_txt,
        }
        
        for file_path in path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in loaders:
                    print(f"Loading {file_path.name}...")
                    docs = loaders[ext](str(file_path))
                    documents.extend(docs)
        
        print(f"\n✓ Loaded {len(documents)} document pages from {directory_path}")
        return documents