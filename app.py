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
from datetime import datetime

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
            analysis_dashboard_tab()
        
        with tab3:
            qa_assistant_tab(genai_analyzer)
        
        with tab4:
            document_library_tab()
    
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
                st.info("Starting document analysis...")
                # Extract text using document processor
                file_content = uploaded_file.getvalue()
                
                try:
                    extraction_result = doc_processor.extract_text(uploaded_file.name, file_content)
                    
                    if not extraction_result['text']:
                        st.error("❌ Could not extract text from the document")
                        return
                    
                    text = extraction_result['text']
                    metadata = extraction_result['metadata']
                    
                    st.success(f"✅ Text extracted successfully from {uploaded_file.type}!")
                    
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
                    
                    # Store the document for Q&A
                    try:
                        st.session_state.current_analysis = {
                            'text': text,
                            'filename': uploaded_file.name,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'metadata': metadata
                        }
                    except Exception as e:
                        st.error(f"Error storing analysis: {str(e)}")
                        st.session_state.current_analysis = {
                            'text': text,
                            'filename': uploaded_file.name,
                            'timestamp': 'Unknown',
                            'metadata': metadata
                        }
                    
                    # Perform ML Analysis
                    if run_ml_analysis and ml_analyzer:
                        with st.spinner("Running ML analysis..."):
                            st.markdown("### 🤖 ML Analysis Results")
                            
                            # Sentiment analysis
                            if include_sentiment:
                                try:
                                    sentiment_result = ml_analyzer.analyze_sentiment(text)
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Sentiment", sentiment_result['sentiment'])
                                    with col2:
                                        st.metric("Polarity", f"{sentiment_result['polarity']:.2f}")
                                    with col3:
                                        st.metric("Subjectivity", f"{sentiment_result['subjectivity']:.2f}")
                                except Exception as e:
                                    st.error(f"Sentiment analysis error: {str(e)}")
                            
                            # Document classification
                            try:
                                doc_type = ml_analyzer.classify_document_type(text)
                                st.markdown(f"**📄 Document Type:** {doc_type['document_type']} (Confidence: {doc_type['confidence']:.1f}%)")
                            except Exception as e:
                                st.error(f"Document classification error: {str(e)}")
                            
                            # Keywords
                            try:
                                keywords = ml_analyzer.extract_keywords(text, 10)
                                if keywords:
                                    st.markdown("**🔑 Key Terms:**")
                                    keyword_text = ", ".join([f"{kw[0]} ({kw[1]:.3f})" for kw in keywords[:8]])
                                    st.write(keyword_text)
                            except Exception as e:
                                st.error(f"Keywords extraction error: {str(e)}")
                            
                            # Named entities
                            if extract_entities:
                                try:
                                    entities = ml_analyzer.extract_named_entities(text)
                                    if any(entities.values()):
                                        st.markdown("**👤 Named Entities:**")
                                        for entity_type, entity_list in entities.items():
                                            if entity_list:
                                                st.write(f"**{entity_type}:** {', '.join(entity_list[:5])}")
                                except Exception as e:
                                    st.error(f"Named entity extraction error: {str(e)}")
                            
                            # Word cloud data
                            if generate_wordcloud:
                                try:
                                    wc_data = ml_analyzer.generate_word_cloud_data(text)
                                    if wc_data.get('word_frequencies'):
                                        st.markdown("**☁️ Most Frequent Words:**")
                                        top_words = list(wc_data['word_frequencies'].items())[:10]
                                        word_text = ", ".join([f"{word} ({count})" for word, count in top_words])
                                        st.write(word_text)
                                except Exception as e:
                                    st.error(f"Word cloud generation error: {str(e)}")
                    
                    # Perform GenAI Analysis
                    if run_genai_analysis and genai_analyzer and genai_analyzer.openai_available:
                        with st.spinner("Running GenAI analysis..."):
                            st.markdown("### 🧠 GenAI Analysis Results")
                            
                            # Summarization
                            if include_summarization:
                                try:
                                    summary_result = genai_analyzer.generate_summary(text, "auto", 150, 50)
                                    if 'summary' in summary_result:
                                        st.markdown("**📝 Summary:**")
                                        st.write(summary_result['summary'])
                                        st.caption(f"Method: {summary_result.get('method', 'unknown')}")
                                except Exception as e:
                                    st.error(f"Summarization error: {str(e)}")
                            
                            # AI Insights
                            try:
                                insights_result = genai_analyzer.generate_insights(text)
                                if insights_result.get('insights'):
                                    st.markdown("**💡 AI Insights:**")
                                    for insight in insights_result['insights'][:3]:
                                        st.write(f"• {insight}")
                            except Exception as e:
                                st.error(f"AI Insights error: {str(e)}")
                    
                    # Show text preview
                    st.markdown("### 📄 Document Preview")
                    preview_length = 2000
                    preview_text = text[:preview_length] + "..." if len(text) > preview_length else text
                    st.text_area("Content:", preview_text, height=300)
                    
                except Exception as e:
                    # Fallback to basic text extraction for unsupported files
                    if uploaded_file.type == "text/plain":
                        try:
                            text = file_content.decode("utf-8")
                            st.success("✅ Text extracted successfully!")
                            
                            # Basic statistics
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
                            
                            # Store for Q&A
                            try:
                                st.session_state.current_analysis = {
                                    'text': text,
                                    'filename': uploaded_file.name,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                            except Exception as e:
                                st.session_state.current_analysis = {
                                    'text': text,
                                    'filename': uploaded_file.name,
                                    'timestamp': 'Unknown'
                                }
                            
                            # Show preview
                            st.markdown("### Document Preview")
                            st.text_area("Content:", text[:2000] + "..." if len(text) > 2000 else text, height=300)
                            
                        except Exception as inner_e:
                            st.error(f"Error extracting text: {str(inner_e)}")
                    else:
                        st.error(f"Error processing {uploaded_file.type} file: {str(e)}")
                        st.info("This file type may require additional processing capabilities.")
                    
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.code(traceback.format_exc())

def analysis_dashboard_tab():
    """Analysis dashboard tab"""
    if 'current_analysis' in st.session_state and st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        st.markdown(f"### Analysis for: {analysis['filename']}")
        st.markdown(f"**Analyzed at:** {analysis['timestamp']}")
        
        # Show basic text statistics
        text = analysis['text']
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
        
        # Show document preview
        st.markdown("### Document Preview")
        st.text_area("Content:", text[:1500] + "..." if len(text) > 1500 else text, height=200)
        
    else:
        st.info("📄 Please upload and analyze a document first to see the analysis dashboard.")

def qa_assistant_tab(genai_analyzer):
    """Q&A Assistant tab"""
    if not ('current_analysis' in st.session_state and st.session_state.current_analysis):
        st.info("📄 Please upload and analyze a document first to use the Q&A assistant.")
        return
    
    analysis = st.session_state.current_analysis
    st.markdown(f"### Q&A for: {analysis['filename']}")
    
    # Check API status and show detailed information
    if genai_analyzer:
        try:
            status = genai_analyzer.check_api_status()
            
            # Show API key status
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            try:
                if hasattr(st, 'secrets') and st.secrets is not None and 'OPENAI_API_KEY' in st.secrets:
                    api_key = st.secrets['OPENAI_API_KEY']
            except Exception:
                # Secrets not available, use environment variable
                pass
            
            if api_key:
                st.info(f"🔑 OpenAI API Key: {'*' * 20}{api_key[-4:] if len(api_key) > 4 else 'FOUND'}")
            else:
                st.error("❌ No OpenAI API Key found")
            
            if status.get('openai_available', False):
                st.success("🧠 OpenAI API is available - Enhanced Q&A enabled")
            else:
                st.warning("⚠️ OpenAI API not available - Using basic keyword search")
                if 'openai_connection' in status:
                    st.error(f"Connection issue: {status['openai_connection']}")
                    
        except Exception as e:
            st.error(f"Error checking API status: {str(e)}")
            st.warning("⚠️ Using basic keyword search")
    
    # Question input
    question = st.text_input(
        "Ask a question about your document:",
        placeholder="e.g., What are the main topics discussed?",
        help="Ask any question about the content of your document"
    )
    
    # Answer method selection
    col1, col2 = st.columns([3, 1])
    with col2:
        method = st.selectbox(
            "Answer Method:",
            ["auto", "openai", "local"],
            help="auto = use best available method"
        )
    
    if question and st.button("🤔 Get Answer", type="primary"):
        with st.spinner("Thinking..."):
            try:
                if genai_analyzer:
                    st.info(f"Asking question: '{question}' using method: '{method}'")
                    
                    # Show some debug info
                    with st.expander("Debug Information", expanded=False):
                        st.write(f"**Question length:** {len(question)} characters")
                        st.write(f"**Document length:** {len(analysis['text'])} characters")
                        st.write(f"**Selected method:** {method}")
                        st.write(f"**GenAI analyzer available:** {genai_analyzer is not None}")
                        
                        if genai_analyzer:
                            st.write(f"**OpenAI available:** {getattr(genai_analyzer, 'openai_available', False)}")
                            st.write(f"**OpenAI client:** {getattr(genai_analyzer, 'openai_client', None) is not None}")
                    
                    result = genai_analyzer.answer_question(analysis['text'], question, method)
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.write(result['answer'])
                    
                    # Show metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                    with col2:
                        st.metric("Method", result.get('method', 'unknown'))
                    
                    if 'error' in result:
                        st.error(f"Note: {result['error']}")
                        
                    # Show full result for debugging
                    with st.expander("Full Result (Debug)", expanded=False):
                        st.json(result)
                        
                else:
                    st.error("Q&A analyzer not available")
                    
            except Exception as e:
                st.error(f"Error answering question: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Suggested questions
    st.markdown("### 🤔 Suggested Questions")
    suggested = [
        "What are the main topics discussed?",
        "What is the purpose of this document?",
        "What are the key findings or conclusions?",
        "Who are the main people or organizations mentioned?",
        "What are the most important points?"
    ]
    
    for suggestion in suggested:
        if st.button(suggestion, key=f"suggest_{hash(suggestion)}"):
            try:
                st.rerun()
            except AttributeError:
                # Fallback for older Streamlit versions
                st.experimental_rerun()

def document_library_tab():
    """Document library tab"""
    st.markdown("### 📚 Document Library")
    
    if 'analyzed_documents' in st.session_state and st.session_state.analyzed_documents:
        st.write(f"You have analyzed {len(st.session_state.analyzed_documents)} documents:")
        
        for i, doc in enumerate(st.session_state.analyzed_documents):
            with st.expander(f"📄 {doc.get('filename', 'Unnamed Document')} - {doc.get('timestamp', 'Unknown time')}"):
                st.write(f"**Type:** {doc.get('type', 'Unknown')}")
                st.write(f"**Size:** {doc.get('size', 'Unknown')} characters")
                if st.button(f"Load Document {i+1}", key=f"load_{i}"):
                    st.session_state.current_analysis = doc
                    st.success("Document loaded! Switch to other tabs to analyze.")
    else:
        st.info("No documents analyzed yet. Upload and analyze documents to see them here.")

if __name__ == "__main__":
    main()
