from typing import List
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

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

        proposition_generator = prompt | self.structured_llm
        
        return proposition_generator
    
    
class PropositionGrader():
    
    def __init__(self, model: str = "llama-3.1-70b-versatile", temperature: float = 0):
        
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(GradePropositions)

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

        proposition_evaluator = prompt | self.structured_llm
        
        return proposition_evaluator