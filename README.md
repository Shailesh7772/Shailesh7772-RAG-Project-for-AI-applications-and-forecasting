
---

# RAG-Project-for-AI-Applications-and-Forecasting

This project is a **Streamlit web application** that leverages **Retrieval-Augmented Generation (RAG)** and **advanced forecasting techniques** to provide intelligent insights, Q\&A capabilities, and trend predictions using unstructured data sources like PDFs.

---

## 🔧 Features

* **Streamlit UI**: Interactive and intuitive web interface for seamless user interaction.
* **Retrieval-Augmented Generation (RAG)**: Integrates LangChain and Sentence Transformers to answer user queries based on uploaded documents.
* **PDF & Unstructured Data Support**: Parses and processes insights from PDFs and various unstructured file formats.
* **Time Series Forecasting**: Uses Prophet for trend and time-based forecasting with support for seasonality and holidays.
* **Interactive Visualizations**: Charts and graphs powered by Plotly for easy interpretation of results.
* **Fast Semantic Search**: Employs FAISS for rapid similarity search using vector embeddings.
* **Modular & Scalable**: Easily adaptable to various AI-driven document-based applications.

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RAG-Project-AI-Forecasting.git
cd RAG-Project-AI-Forecasting
```

### 2. Install Dependencies

You can install everything using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install streamlit langchain sentence-transformers faiss-cpu plotly pandas numpy
pip install prophet cmdstanpy holidays
pip install "unstructured[pdf]"
```

---

## 🚀 Usage

### 1. Set Environment Variables

Create a `.env` file in the project root to include any API keys or configurations (if needed).

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

> Replace `app.py` with your main application file if it's named differently.

### 3. Interact with the App

* Upload documents (PDFs or unstructured text)
* Ask questions related to content
* View extracted insights and time-series forecasts

---

## 📦 Dependencies

* `streamlit`: Web application interface
* `langchain`, `langchain_community`, `langchain_core`: RAG and LLM tools
* `sentence-transformers`: Embedding models
* `unstructured[pdf]`: Document and PDF parsing
* `prophet`, `cmdstanpy`, `holidays`: Time series modeling
* `plotly`: Visualization library
* `pandas`, `numpy`: Data manipulation
* `faiss-cpu`: Vector similarity search

---

## 📄 License

This project is licensed under the **MIT License**.

---

