from typing import List
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

import os

from transcripts_rag.ingestion import dataloader, doc_split


# Data model
class GeneratePropositions(BaseModel):
    """List of all the propositions in a given document"""

    propositions: List[str] = Field(
        description="List of propositions (factual, self-contained, and concise information)"
    )
    
# Data model
class GradePropositions(BaseModel):
    """Grade a given proposition on accuracy, clarity, completeness, and conciseness"""

    accuracy: int = Field(
        description="Rate from 1-10 based on how well the proposition reflects the original text."
    )
    
    clarity: int = Field(
        description="Rate from 1-10 based on how easy it is to understand the proposition without additional context."
    )

    completeness: int = Field(
        description="Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers)."
    )

    conciseness: int = Field(
        description="Rate from 1-10 based on whether the proposition is concise without losing important information."
    )


class PropositionGenerator():
    
    def __init__(self, model: str = "llama-3.1-70b-versatile", temperature: float = 0):
        
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(GeneratePropositions)
        self.proposition_generator = None
        self.propositions = []

    def generate_propositions(self):
    # Few shot prompting --- We can add more examples to make it good
        proposition_examples = [
            {"document": 
                "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.", 
            "propositions": 
                "['Neil Armstrong was an astronaut.', 'Neil Armstrong walked on the Moon in 1969.', 'Neil Armstrong was the first person to walk on the Moon.', 'Neil Armstrong walked on the Moon during the Apollo 11 mission.', 'The Apollo 11 mission occurred in 1969.']"
            },
        ]

        example_proposition_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{document}"),
                ("ai", "{propositions}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt = example_proposition_prompt,
            examples = proposition_examples,
        )

        # Prompt
        system = """Please break down the following text into simple, self-contained propositions. Ensure that each proposition meets the following criteria:

            1. Express a Single Fact: Each proposition should state one specific fact or claim.
            2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
            3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
            4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
            5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses."""
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                few_shot_prompt,
                ("human", "{document}"),
            ]
        )

        self.proposition_generator = prompt | self.structured_llm
        
    def get_propositions(self, docs: list):
        
        self.propositions = []
                
        for i in tqdm(range(len(docs))):
            
            doc = docs[i]
            file_name = doc.metadata['file_name']
            file_path = doc.metadata['file_path']
            creation_date = doc.metadata['creation_date']
            last_modified_date = doc.metadata['last_modified_date']
            split_ls = doc.metadata['file_name'].rstrip('.txt').split("_")
            
            
            response = self.proposition_generator.invoke({"document": doc.page_content}) # Creating proposition
            for proposition in response.propositions:
                self.propositions.append(Document(proposition, 
                                                  metadata={ 'file_path': file_path,
                    'file_name': file_name,
                    'file_type': 'text/plain',
                    'file_size': 50540,
                    'creation_date': creation_date,
                    'last_modified_date': last_modified_date, "chunk_id": i+1,
                    'ticker': split_ls[-1],
                    'year': int(split_ls[0]),
                    'quarter': int(split_ls[1])
                    }))
                
            
    
class PropositionGrader():
    
    def __init__(self, model: str = "llama-3.1-70b-versatile", temperature: float = 0):
        
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(GradePropositions)
        self.proposition_evaluator = None
        self.treshold_dict = {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7}
        self.evaluated_propositions = []

    def grade_propositions(self):
        
        evaluation_prompt_template = """
            Please evaluate the following proposition based on the criteria below:
            - **Accuracy**: Rate from 1-10 based on how well the proposition reflects the original text.
            - **Clarity**: Rate from 1-10 based on how easy it is to understand the proposition without additional context.
            - **Completeness**: Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers).
            - **Conciseness**: Rate from 1-10 based on whether the proposition is concise without losing important information.

            Example:
            Docs: In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.

            Propositons_1: Neil Armstrong was an astronaut.
            Evaluation_1: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

            Propositons_2: Neil Armstrong walked on the Moon in 1969.
            Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

            Propositons_3: Neil Armstrong was the first person to walk on the Moon.
            Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

            Propositons_4: Neil Armstrong walked on the Moon during the Apollo 11 mission.
            Evaluation_4: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

            Propositons_5: The Apollo 11 mission occurred in 1969.
            Evaluation_5: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

            Format:
            Proposition: "{proposition}"
            Original Text: "{original_text}"
            """
        prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", evaluation_prompt_template),
                    ("human", "{proposition}, {original_text}"),
                ]
            )

        self.proposition_evaluator = prompt | self.structured_llm
        
    def evaluate_propositions(self, proposition: list, original_text: str):
        
        response = self. proposition_evaluator.invoke({"proposition": proposition, "original_text": original_text})
    
        # Parse the response to extract scores
        scores = {"accuracy": response.accuracy, "clarity": response.clarity, "completeness": response.completeness, "conciseness": response.conciseness}  # Implement function to extract scores from the LLM response
        return scores
    
    def passes_quality_check(self, scores):
        for category, score in scores.items():
            if score < self.treshold_dict[category]:
                return False
        return True
    
    def generate_evaluations(self, propositions: List[Document], doc_splits: List[Document]):
        
        self.evaluated_propositions = []
        
        for idx, proposition in tqdm(enumerate(propositions)):
            scores = self.evaluate_propositions(proposition.page_content, doc_splits[proposition.metadata['chunk_id'] - 1].page_content)
            if self.passes_quality_check(scores):
                # Proposition passes quality check, keep it
                self.evaluated_propositions.append(proposition)
                print(f"Success: {idx+1}) Propostion: {proposition.page_content} \n Scores: {scores}")
            else:
                # Proposition fails, discard or flag for further review
                print(f"Fail: {idx+1}) Propostion: {proposition.page_content} \n Scores: {scores}")
                

if __name__ == '__main__':
    
    try:
    
        doc = dataloader('data/')
        doc = doc_split(doc)
        
        #instantiate PropositionGenerator
        prop_generator = PropositionGenerator(model='gemma2-9b-it')
        
        #Generate propositions
        prop_generator.generate_propositions()
        
        #get_propositions
        prop_generator.get_propositions(docs=doc)
            
        prop_evaluator = PropositionGrader(model='gemma2-9b-it')
        prop_evaluator.grade_propositions()
        
        prop_evaluator.generate_evaluations(propositions=prop_generator.propositions, doc_splits=doc)
        
        embedding_model = OllamaEmbeddings(model='nomic-embed-text:v1.5', show_progress=True)
        

        vectorstore_propositions = FAISS.from_documents(prop_evaluator.evaluated_propositions, embedding_model)
        
        if os.path.exists("faiss_transcript_index"):
            vectorstore_propositions.save_local("faiss_transcript_index")
        
        else:
            old_vectorstore_propositions = FAISS.load_local(
                "faiss_transcript_index", embedding_model, allow_dangerous_deserialization=True
            )
            
            old_vectorstore_propositions.merge_from(vectorstore_propositions)
            
            vectorstore_propositions.save_local("faiss_transcript_index")
            
            import ipdb;ipdb.set_trace()
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
    
    
    