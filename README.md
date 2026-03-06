# 🏎️ FIA Race Control AI (2026 Regulations)

An AI-powered RAG (Retrieval-Augmented Generation) bot designed to instantly answer questions based strictly on the massive 2026 Formula 1 Sporting and Technical Regulations.

##  The Tech Stack
* **Architecture:** Retrieval-Augmented Generation (RAG)
* **LLM (Brain):** Google Gemini 2.5 Flash 
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local processing)
* **Vector Database:** ChromaDB
* **Frontend:** Streamlit

##  How it Works
Formula 1 rulebooks are incredibly dense, containing over 300 pages of legal jargon and CAD coordinates. This tool acts as an intelligent search engine and translator. 

It uses a strict "Closed-Loop" security prompt. When a user asks a question, the local embedding model retrieves the 4 most relevant paragraphs from the official PDFs. Gemini then reads *only* those paragraphs to generate a plain-English summary. If the answer isn't in the rulebook, the AI is hard-coded to refuse to answer, entirely preventing hallucinations.

##  Run it Locally
1. Clone this repository.
2. Create a virtual environment and run `pip install -r requirements.txt`.
3. Add your Gemini API key to an `env/.env` file.
4. Run `streamlit run src/app.py`.