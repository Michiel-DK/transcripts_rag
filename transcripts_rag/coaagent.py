from typing import List
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex
import faiss
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import os

from transcripts_rag.preprocessing import TextCleaner

import llama_index

class CoaAgent:
    def __init__(self, llm, embedding):
        self.llm = llm
        self.embedding = embedding
        
    def setup_faiss(self, embedding_dimension:int = 512):
        
        fais_index = faiss.IndexFlatL2(embedding_dimension)
        vector_store = FaissVectorStore(faiss_index=fais_index)
        

class FaissVectorStoreAgent:
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = None
        self.pipeline = None
        self.tool_list = None
        
    def setup_faiss(self, embedding_dimension:int = 512):
        
        fais_index = faiss.IndexFlatL2(embedding_dimension)
    
        self.vector_store = FaissVectorStore(faiss_index=fais_index)
                
    def integestion_pipe(self):
        self.pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            TextCleaner()
        ],
        vector_store=self.vector_store,
        )
        
    def generate_tools(self, base_path, path_list:List[str]):
        
        docs_list = []
        
        self.tool_list = []
        
        full_path_list = [base_path+path for path in path_list]
                
        #name_list = [x.split('_')[-1].split('.')[0] for x in path_list]
        name_list = [x.split('_')[0] for x in path_list]
                
        for path in full_path_list:
            doc = SimpleDirectoryReader(
                input_files=[path]
            ).load_data()
                        
            docs_list.append(doc)
            
        for doc, name in zip(docs_list, name_list):
            nodes = self.pipeline.run(documents=doc)
                        
            index = VectorStoreIndex(nodes= nodes, embed_model=self.embedding)
                        
            index.storage_context.persist(persist_dir=f"storage/{name}")
            engine = index.as_query_engine(similarity_top_k=3)
                                    
            tool = QueryEngineTool(
                    query_engine=engine,
                    metadata=ToolMetadata(
                        name=f"{name}",
                        description=(
                            f"Provides information about {name} financials. "
                            "Use a detailed plain text question as input to the tool. "
                            "The input is used to power a semantic search engine."
                        ),
                    ),
                )
                        
            self.tool_list.append(tool)
            
        
        return self.tool_list
        
            
if __name__ == '__main__':
    
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        embedding_model =  OllamaEmbedding(model_name="nomic-embed-text:v1.5")
        
        fva = FaissVectorStoreAgent(embedding=embedding_model)
        fva.setup_faiss()
        fva.integestion_pipe()
        
        check_ls = os.listdir('data/10k/')[-2:]
        
        query_engine_tools = fva.generate_tools('data/10k/', check_ls)
                
        from llama_index.packs.agents_coa import CoAAgentPack

        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        pack = CoAAgentPack(tools=query_engine_tools, llm=llm)
        
        import nest_asyncio
        nest_asyncio.apply()
        response = pack.run("How did lyft revenue growth compare to uber?")
        
            
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)