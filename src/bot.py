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
        
        system_msg = """You are the FIA Race Control AI.
        You must answer the user's question ONLY using the technical or sporting context provided below.
        If the answer is not in the context, strictly say: 'I cannot find this information in the rulebooks.'
        Do not use your own knowledge about F1.
        
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