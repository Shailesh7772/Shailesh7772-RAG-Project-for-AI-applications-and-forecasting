# vectors.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class EmbeddingsManager:
    def __init__(self):
        # Use the faster MiniLM model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        # Larger chunks, less overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks
            chunk_overlap=20,  # Minimal overlap
            separators=["\n\n", "\n", ". ", "! ", "? ", ", "],  # More specific separators
            length_function=len
        )

    def batch_process_chunks(self, chunks, batch_size=32):
        """Process chunks in batches"""
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    def create_embeddings(self, pdf_files):
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize
            status_text.text("Initializing...")
            progress_bar.progress(0)
            
            if not os.path.exists("faiss_index"):
                os.makedirs("faiss_index")
            
            # Step 2: Process PDFs
            status_text.text("Reading PDFs...")
            all_chunks = []
            for i, pdf_path in enumerate(pdf_files):
                progress = (i + 1) / len(pdf_files) * 0.3
                progress_bar.progress(progress)
                status_text.text(f"Reading PDF {i+1} of {len(pdf_files)}...")
                
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                chunks = self.text_splitter.split_documents(pages)
                all_chunks.extend(chunks)
            
            # Step 3: Create Embeddings
            total_chunks = len(all_chunks)
            status_text.text(f"Creating embeddings for {total_chunks} chunks...")
            progress_bar.progress(0.4)
            
            vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            
            # Step 4: Save Index
            status_text.text("Saving index...")
            progress_bar.progress(0.9)
            vectorstore.save_local("faiss_index")
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            return f"✅ Processed {total_chunks} chunks"
            
        except Exception as e:
            status_text.text("❌ Error occurred")
            progress_bar.empty()
            raise Exception(f"Error: {str(e)}")

    def process_pdf(self, pdf_path):
        """Process a single PDF file"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return self.text_splitter.split_documents(pages)