"""
Smart Document Analyzer - Streamlit Web Application (Simplified for Deployment)
A comprehensive ML/GenAI-powered document analysis platform

Author: Rishav Kant
Institution: Birla Institute of Technology, Mesra
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import traceback

# Configure page
st.set_page_config(
    page_title="Smart Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_documents' not in st.session_state:
    st.session_state.analyzed_documents = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

def check_imports():
    """Check which packages are available"""
    available_packages = {}
    
    # Check core packages
    packages_to_check = [
        'plotly', 'nltk', 'textblob', 'wordcloud', 
        'PyPDF2', 'docx', 'openpyxl', 'openai', 'transformers'
    ]
    
    for package in packages_to_check:
        try:
            if package == 'docx':
                import docx
            else:
                __import__(package)
            available_packages[package] = True
        except ImportError:
            available_packages[package] = False
    
    return available_packages

def safe_import_analyzers():
    """Safely import analyzer modules"""
    try:
        # Add src directory to path for imports
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from utils.document_processor import DocumentProcessor
        from models.ml_analyzer import MLDocumentAnalyzer
        from models.genai_analyzer import GenAIDocumentAnalyzer
        
        doc_processor = DocumentProcessor()
        ml_analyzer = MLDocumentAnalyzer()
        genai_analyzer = GenAIDocumentAnalyzer()
        
        return doc_processor, ml_analyzer, genai_analyzer, True
    except Exception as e:
        st.error(f"Error importing analyzers: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None, False

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📄 Smart Document Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by ML & GenAI | Built by Rishav Kant, BIT Mesra**")
    
    # Check package availability
    available_packages = check_imports()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x150/1f77b4/white?text=Smart+Document+Analyzer", 
                caption="ML + GenAI Document Analysis")
        
        st.markdown("### 🚀 Features")
        st.markdown("""
        - **Document Processing**: PDF, DOCX, TXT, CSV
        - **Traditional ML**: Classification, Clustering, NLP
        - **GenAI Integration**: GPT-powered insights
        - **Sentiment Analysis**: Advanced emotion detection
        - **Q&A System**: Ask questions about your documents
        - **Visualization**: Interactive charts and word clouds
        """)
        
        st.markdown("### 📊 Package Status")
        for package, available in available_packages.items():
            status = "✅" if available else "❌"
            st.write(f"{status} {package}")
    
    # Try to load analyzers
    doc_processor, ml_analyzer, genai_analyzer, success = safe_import_analyzers()
    
    if not success:
        st.error("⚠️ Some analyzer modules could not be loaded. The app is running in basic mode.")
        st.info("This is likely due to missing dependencies or import issues during deployment.")
        
        # Basic file upload functionality
        st.markdown('<h2 class="sub-header">Basic Document Upload</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a document to analyze",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx'],
            help="Supported formats: PDF, DOCX, TXT, CSV, XLSX"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**Size:** {uploaded_file.size:,} bytes")
            st.markdown(f"**Type:** {uploaded_file.type}")
            
            # Show file contents for text files
            if uploaded_file.type == "text/plain":
                try:
                    content = uploaded_file.getvalue().decode("utf-8")
                    st.text_area("File Content Preview:", content[:1000] + "..." if len(content) > 1000 else content, height=200)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    else:
        st.success("✅ All analysis models loaded successfully!")
        
        # Main tabs with full functionality
        tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "📈 Analysis Dashboard", "💬 Q&A Assistant", "📚 Document Library"])
        
        with tab1:
            upload_and_analyze_tab(doc_processor, ml_analyzer, genai_analyzer)
        
        with tab2:
            st.info("Analysis dashboard will be available after uploading and analyzing a document.")
        
        with tab3:
            st.info("Q&A assistant will be available after uploading and analyzing a document.")
        
        with tab4:
            st.info("Document library will show previously analyzed documents.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <hr>
        <p><strong>Smart Document Analyzer</strong> | Developed by Rishav Kant | Birla Institute of Technology, Mesra</p>
        <p>Combining Traditional ML with Modern GenAI for Comprehensive Document Analysis</p>
        <p><a href="https://github.com/rishav1708/Smardy" target="_blank">🔗 View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

def upload_and_analyze_tab(doc_processor, ml_analyzer, genai_analyzer):
    """Document upload and analysis tab"""
    st.markdown('<h2 class="sub-header">Upload Your Document</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a document to analyze",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx'],
            help="Supported formats: PDF, DOCX, TXT, CSV, XLSX"
        )
        
        if uploaded_file:
            # Analysis options
            st.markdown("### Analysis Options")
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                run_ml_analysis = st.checkbox("Traditional ML Analysis", value=True)
                include_sentiment = st.checkbox("Sentiment Analysis", value=True)
            
            with col_opt2:
                run_genai_analysis = st.checkbox("GenAI Analysis", value=False)
                include_summarization = st.checkbox("Summarization", value=False)
            
            with col_opt3:
                extract_entities = st.checkbox("Named Entity Extraction", value=True)
                generate_wordcloud = st.checkbox("Word Cloud", value=True)
    
    with col2:
        if uploaded_file:
            st.markdown("### Document Info")
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**Size:** {uploaded_file.size:,} bytes")
            st.markdown(f"**Type:** {uploaded_file.type}")
    
    # Analyze button
    if uploaded_file and st.button("🚀 Analyze Document", type="primary"):
        with st.spinner("Processing document..."):
            try:
                # Extract text
                file_content = uploaded_file.getvalue()
                
                # Basic text extraction
                if uploaded_file.type == "text/plain":
                    text = file_content.decode("utf-8")
                    st.success("✅ Text extracted successfully!")
                    
                    # Display basic statistics
                    st.markdown("### Document Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Characters", len(text))
                    with col2:
                        word_count = len(text.split())
                        st.metric("Words", word_count)
                    with col3:
                        sentence_count = len(text.split('.'))
                        st.metric("Sentences", sentence_count)
                    with col4:
                        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
                        st.metric("Avg Word Length", f"{avg_word_length:.1f}")
                    
                    # Show text preview
                    st.markdown("### Document Preview")
                    st.text_area("Content:", text[:2000] + "..." if len(text) > 2000 else text, height=300)
                    
                else:
                    st.info("Advanced document processing requires additional modules to be properly installed.")
                    st.info("Currently showing basic file information only.")
                    
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
