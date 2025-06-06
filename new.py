import streamlit as st
import time
import base64
import os
import pandas as pd
import shutil

# Set page config at the very top
st.set_page_config(
    page_title="RAG in Real Estate",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from vectors import EmbeddingsManager
    from chatbot import ChatbotManager
    from predictor import RealEstatePredictor
except ImportError:
    st.error("Required packages are not installed. Please check requirements.txt")
    st.stop()

# Initialize session state
if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'temp_paths' not in st.session_state:
    st.session_state['temp_paths'] = []
if 'error_log' not in st.session_state:
    st.session_state['error_log'] = []
if 'embeddings_created' not in st.session_state:
    st.session_state['embeddings_created'] = False

def validate_file(file):
    """Validate uploaded files"""
    try:
        if file.name.endswith('.pdf'):
            if file.size > 10 * 1024 * 1024:  # 10MB limit
                return False, "PDF file too large (max 10MB)"
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
            required_columns = ['RegionName', 'RegionID']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"CSV missing required columns: {', '.join(missing_columns)}"
            file.seek(0)  # Reset file pointer
        return True, "File valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        temp_dir = "temp_files"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        return True, "Cleanup successful"
    except Exception as e:
        return False, f"Cleanup error: {str(e)}"

def displayPDF(uploaded_file):
    """Display uploaded PDF file"""
    try:
        # Create a temporary directory if it doesn't exist
        if not os.path.exists('temp_pdf'):
            os.makedirs('temp_pdf')
            
        # Save PDF to temporary file
        temp_path = os.path.join('temp_pdf', uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Display file info
        file_size = os.path.getsize(temp_path) / 1024  # KB
        st.write(f"File: {uploaded_file.name} ({file_size:.1f} KB)")
        
        # Add download button
        with open(temp_path, "rb") as f:
            st.download_button(
                " Download PDF",
                f,
                file_name=uploaded_file.name,
                mime="application/pdf"
            )
            
        # Display PDF using iframe
        st.markdown(f"""
            <iframe
                src="data:application/pdf;base64,{base64.b64encode(open(temp_path, 'rb').read()).decode('utf-8')}"
                width="100%"
                height="800px"
                style="border: none; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1);"
            >
            </iframe>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        st.write("Please use the download button to view the file.")

def save_uploaded_file(uploaded_file, temp_dir="temp_files"):
    """Save uploaded file and return path"""
    try:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def handle_uploaded_file(uploaded_file):
    """Simple file handling without preview or download"""
    try:
        # Display file info
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        st.write(f"📄 {uploaded_file.name} ({file_size:.1f} KB)")
        
        # Save file and return path
        return save_uploaded_file(uploaded_file)
        
    except Exception as e:
        st.error(f"Error handling file: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.title("🏡 RAG in Real Estate")
    st.markdown("---")
    st.markdown("""
    ### Features:
    📄 PDF Document Analysis  
    💬 Smart Q&A System  
    📈 Price Predictions  
    """)
    
    # Cleanup and Reset buttons
    if st.button("🧹 Cleanup Files"):
        success, message = cleanup_temp_files()
        if success:
            st.success(message)
        else:
            st.error(message)
            
    if st.button("🔄 Reset Session"):
        # Delete the faiss_index directory
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Error log
    if st.session_state['error_log']:
        with st.expander("⚠️ Error Log"):
            for error in st.session_state['error_log']:
                st.error(error)

# Main content
st.title("RAG in Real Estate")

# Create three columns with better proportions
col1, col2, col3 = st.columns([1, 2, 1])

# Column 1: File Upload
with col1:
    st.header("📂 Upload")
    uploaded_files = st.file_uploader(
        "Upload Files",
        type=["pdf", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} File(s) uploaded")
        for uploaded_file in uploaded_files:
            with st.expander(f"📎 {uploaded_file.name}", expanded=True):
                temp_path = handle_uploaded_file(uploaded_file)
                if temp_path:
                    st.session_state['temp_paths'].append(temp_path)

# Column 2: Processing (Wider)
with col2:
    st.header("🧠 Processing")
    if uploaded_files:
        # Separate PDF and CSV files
        pdf_files = [f for f in uploaded_files if f.name.endswith('.pdf')]
        csv_files = [f for f in uploaded_files if f.name.endswith('.csv')]
        
        # Handle PDFs - RAG Processing
        if pdf_files:
            st.subheader("📄 PDF Processing")
            
            # Show embedding status
            if st.session_state['embeddings_created']:
                st.success("✅ Embeddings ready - Ask questions in the chat!")
            else:
                create_embeddings = st.checkbox("Create Embeddings for Q&A")
                if create_embeddings:
                    try:
                        embeddings_manager = EmbeddingsManager()
                        result = embeddings_manager.create_embeddings(
                            [f for f in st.session_state['temp_paths'] if f.endswith('.pdf')]
                        )
                        
                        st.session_state['chatbot_manager'] = ChatbotManager()
                        st.session_state['embeddings_created'] = True
                        st.success("✅ Embeddings created! Ready for questions.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.write("Debug info:")
                        st.write("PDF paths:", [f for f in st.session_state['temp_paths'] if f.endswith('.pdf')])
        
        # Handle CSV - Prediction Processing
        if csv_files:
            st.subheader("📈 Price Prediction")
            try:
                csv_file = csv_files[0]
                df = pd.read_csv(csv_file)
                
                # Region selection
                regions = sorted(df['RegionName'].unique())
                selected_region = st.selectbox("Select Region", regions)
                periods = st.slider("Forecast Months", 3, 24, 12)
                
                if st.button("Generate Forecast"):
                    with st.spinner("Generating predictions..."):
                        predictor = RealEstatePredictor()
                        predictor.train(df, selected_region)
                        
                        # Get and display model metrics
                        metrics = predictor.evaluate_model()
                        st.subheader("📊 Model Performance")
                        mcol1, mcol2 = st.columns(2)
                        with mcol1:
                            st.metric("MAPE", metrics['MAPE'])
                            st.metric("RMSE", metrics['RMSE'])
                        with mcol2:
                            st.metric("Data Points", metrics['Data Points'])
                            st.metric("Training Period", metrics['Training Period'])
                        
                        # Generate forecast
                        forecast, fig = predictor.predict(periods)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Format and display forecast table
                        forecast_df = forecast.copy()
                        forecast_df.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
                        forecast_df = forecast_df.round(2)
                        st.dataframe(forecast_df)
                        
                        # Download option
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Forecast",
                            data=csv,
                            file_name=f"forecast_{selected_region}.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

# Column 3: Chat Interface (Narrower)
with col3:
    st.header("💬 Chat")
    if uploaded_files and st.session_state['embeddings_created']:
        # Display chat history
        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question"):
            st.session_state['messages'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Add a spinner to show processing status
            with st.spinner("Thinking..."):
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    response = st.session_state['chatbot_manager'].get_response(prompt)
                    message_placeholder.markdown(response)
                    st.session_state['messages'].append({"role": "assistant", "content": response})
    else:
        st.info("📝 Please upload PDFs and create embeddings first")