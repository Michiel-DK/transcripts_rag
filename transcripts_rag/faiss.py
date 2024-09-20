from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os

class CustomFaissVectorStore():
    
    def __init__(self, ollama_embeddings: str = 'nomic-embed-text:v1.5', index_path: str = 'faiss_index'):
        
        self.embeddings_model = OllamaEmbeddings(model=ollama_embeddings, show_progress=True)
        self.index_path = index_path
        self.vector_store = None
        
    def create_vector_store(self, docs):
        self.vector_store = FAISS.from_documents(docs, self.embeddings_model)
        
    def save_index(self, path:str = 'faiss_index'):
        
        if not os.path.exists(path):
            self.vector_store.save_local(path)
                
        else:
            old_vectorstore_propositions = FAISS.load_local(
                    path, self.embeddings_model, allow_dangerous_deserialization=True
                )
                
            old_vectorstore_propositions.merge_from(self.vector_store)
                
            self.vector_store.save_local(path)
            
    def load_index(self):
        self.vector_store = FAISS.load_local(
            self.index_path, self.embeddings_model, allow_dangerous_deserialization=True
        )
