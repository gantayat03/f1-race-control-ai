import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv("env/.env")

class RaceControlBot:
    def __init__(self):
        print("Initializing F1 Race Control Bot...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings
        )
        
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        system_msg = """You are the FIA Race Control AI, an expert in Formula 1 regulations.
        Your job is to read the provided CONTEXT and answer the user's question intelligently.
        
        RULES FOR ANSWERING:
        1. The rulebook uses dense legal jargon. You must act as a translator for the fan.
        2. Understand the intent of the user's question. You do not need an exact word-for-word match. Synthesize the facts.
        3. If the context describes a rule but doesn't list a specific time penalty (e.g., "it will be reported to the stewards"), explain exactly what the context says instead of saying you can't find it.
        4. You must strictly ground your facts in the CONTEXT. Do not use outside knowledge.
        5. ONLY if the CONTEXT is completely unrelated to the question, say: 'I cannot find this information in the rulebooks.'
        
        CONTEXT:
        {context}"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{question}")
        ])

    def ask(self, query):
        print(f"\n[Searching Rulebooks for: '{query}']...")
        docs = self.db.similarity_search(query, k=4)
    
        print("\nRAW CHUNKS RETRIEVED FROM DATABASE:")
        for i, doc in enumerate(docs):
            print(f"Chunk {i+1}: {doc.page_content[:200]}...\n")
        print("----------------------------------------------\n")
        
        context_text = "\n\n".join([doc.page_content for doc in docs])
    
        final_prompt = self.prompt.format_messages(
            context=context_text, 
            question=query
        )
        
        response = self.llm.invoke(final_prompt)
        return response.content

#test script
if __name__ == "__main__":
    bot = RaceControlBot()

    test_question = "What is the penalty for speeding in the pit lane during a race?"
    answer = bot.ask(test_question)
    
    print("\n FIA RACE CONTROL DECISION:")
    print(answer)