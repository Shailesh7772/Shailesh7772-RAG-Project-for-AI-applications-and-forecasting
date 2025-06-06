# chatbot.py

import os
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import time

class ChatbotManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.llm = Ollama(
            model="llama3.2",
            temperature=0.5,
            num_ctx=2048,
            top_k=10,
            top_p=0.9,
            repeat_penalty=1.1
        )
        self.qa_chain = None
        self.setup_qa_chain()

    def setup_qa_chain(self):
        """Initialize the QA chain with the saved FAISS index"""
        try:
            vector_store = FAISS.load_local(
                "faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
        except Exception as e:
            raise Exception(f"Error setting up QA chain: {str(e)}")

    def get_response(self, query, timeout=60):
        """Get response with timeout"""
        try:
            if not self.qa_chain:
                return "Error: QA system not initialized. Please process PDFs first."
            
            start_time = time.time()
            
            formatted_query = f"Based on the document, {query}"
            
            response = self.qa_chain({"query": formatted_query})
            
            if time.time() - start_time > timeout:
                return "Response took too long. Please try a more specific question."
            
            if 'result' in response:
                return response['result']
            else:
                return "No relevant information found. Please try rephrasing your question."
            
        except Exception as e:
            return f"Error: Could not generate response. Please try again or rephrase your question. ({str(e)})"

    def format_response(self, text):
        """Format the response for better readability"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Add bullet points for lists
        if ',' in text and len(text) > 100:
            items = [item.strip() for item in text.split(',')]
            if len(items) > 2:
                return "• " + "\n• ".join(items)
        
        return text