"""
Smart Document Analyzer - Streamlit Web Application
A comprehensive ML/GenAI-powered document analysis platform

Author: Rishav Kant
Institution: Birla Institute of Technology, Mesra
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import io
import base64
from datetime import datetime
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.document_processor import DocumentProcessor
from models.ml_analyzer import MLDocumentAnalyzer
from models.genai_analyzer import GenAIDocumentAnalyzer

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
    .insight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bbdefb;
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

# Initialize analyzers
@st.cache_resource
def load_analyzers():
    """Load and cache the analysis models"""
    doc_processor = DocumentProcessor()
    ml_analyzer = MLDocumentAnalyzer()
    genai_analyzer = GenAIDocumentAnalyzer()
    return doc_processor, ml_analyzer, genai_analyzer

# Load analyzers
try:
    doc_processor, ml_analyzer, genai_analyzer = load_analyzers()
    st.success("✅ All analysis models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📄 Smart Document Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by ML & GenAI | Built by Rishav Kant, BIT Mesra**")
    
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
        
        st.markdown("### 📊 Model Status")
        status = genai_analyzer.check_api_status()
        st.write("🤖 Local ML Models: ✅" if all(status['local_models'].values()) else "🤖 Local ML Models: ⚠️")
        st.write("🧠 OpenAI API: ✅" if status['openai_available'] else "🧠 OpenAI API: ❌")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "📈 Analysis Dashboard", "💬 Q&A Assistant", "📚 Document Library"])
    
    with tab1:
        upload_and_analyze_tab()
    
    with tab2:
        analysis_dashboard_tab()
    
    with tab3:
        qa_assistant_tab()
    
    with tab4:
        document_library_tab()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <hr>
        <p><strong>Smart Document Analyzer</strong> | Developed by Rishav Kant | Birla Institute of Technology, Mesra</p>
        <p>Combining Traditional ML with Modern GenAI for Comprehensive Document Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def upload_and_analyze_tab():
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
                run_genai_analysis = st.checkbox("GenAI Analysis", value=True)
                include_summarization = st.checkbox("Summarization", value=True)
            
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
                extraction_result = doc_processor.extract_text(uploaded_file.name, file_content)
                
                if not extraction_result['text']:
                    st.error("❌ Could not extract text from the document")
                    return
                
                text = extraction_result['text']
                metadata = extraction_result['metadata']
                
                # Perform analyses
                analysis_results = {
                    'filename': uploaded_file.name,
                    'timestamp': datetime.now(),
                    'text': text,
                    'metadata': metadata,
                    'document_stats': doc_processor.get_document_stats(text)
                }
                
                # Traditional ML Analysis
                if run_ml_analysis:
                    with st.spinner("Running ML analysis..."):
                        analysis_results.update({
                            'classification': ml_analyzer.classify_document_type(text),
                            'keywords': ml_analyzer.extract_keywords(text, 15),
                            'readability': ml_analyzer.analyze_readability(text),
                            'word_cloud_data': ml_analyzer.generate_word_cloud_data(text)
                        })
                        
                        if extract_entities:
                            analysis_results['entities'] = ml_analyzer.extract_named_entities(text)
                        
                        if include_sentiment:
                            analysis_results['sentiment'] = ml_analyzer.analyze_sentiment(text)
                
                # GenAI Analysis
                if run_genai_analysis:
                    with st.spinner("Running GenAI analysis..."):
                        if include_summarization:
                            analysis_results['summary'] = genai_analyzer.generate_summary(text)
                        
                        analysis_results['genai_insights'] = genai_analyzer.generate_insights(text)
                        analysis_results['genai_keywords'] = genai_analyzer.generate_keywords_genai(text)
                
                # Store results
                st.session_state.current_analysis = analysis_results
                st.session_state.analyzed_documents.append(analysis_results)
                
                st.success("✅ Analysis completed successfully!")
                
                # Display quick results
                display_analysis_summary(analysis_results)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

def display_analysis_summary(results):
    """Display a summary of analysis results"""
    st.markdown('<h3 class="sub-header">📊 Analysis Summary</h3>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", f"{results['document_stats']['word_count']:,}")
    
    with col2:
        if 'sentiment' in results:
            sentiment = results['sentiment']['sentiment']
            st.metric("Sentiment", sentiment, delta=f"{results['sentiment']['confidence']:.2f}")
        else:
            st.metric("Sentences", results['document_stats']['sentence_count'])
    
    with col3:
        if 'classification' in results:
            doc_type = results['classification']['document_type']
            st.metric("Document Type", doc_type)
        else:
            st.metric("Paragraphs", results['document_stats']['paragraph_count'])
    
    with col4:
        if 'readability' in results:
            difficulty = results['readability']['difficulty_level']
            st.metric("Readability", difficulty)
        else:
            st.metric("Characters", f"{results['document_stats']['character_count']:,}")
    
    # Summary text
    if 'summary' in results:
        st.markdown("### 📝 Document Summary")
        st.markdown(f"*Generated using {results['summary']['method']}*")
        st.info(results['summary']['summary'])
    
    # Top keywords
    if 'keywords' in results:
        st.markdown("### 🔑 Top Keywords")
        keywords_df = pd.DataFrame(results['keywords'], columns=['Keyword', 'Score'])
        st.dataframe(keywords_df.head(10), use_container_width=True)

def analysis_dashboard_tab():
    """Analysis dashboard with visualizations"""
    if not st.session_state.current_analysis:
        st.info("👆 Please upload and analyze a document first")
        return
    
    results = st.session_state.current_analysis
    
    st.markdown('<h2 class="sub-header">📈 Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    # Document overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Document Overview")
        
        # Create metrics visualization
        stats = results['document_stats']
        metrics_data = {
            'Metric': ['Words', 'Sentences', 'Paragraphs', 'Unique Words'],
            'Count': [stats['word_count'], stats['sentence_count'], 
                     stats['paragraph_count'], stats['unique_words']]
        }
        
        fig_metrics = px.bar(
            x=metrics_data['Metric'],
            y=metrics_data['Count'],
            title="Document Statistics",
            color=metrics_data['Count'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        
        if 'sentiment' in results:
            sentiment_data = results['sentiment']
            st.markdown(f"""
            <div class="metric-card">
                <strong>Sentiment:</strong> {sentiment_data['sentiment']}<br>
                <strong>Polarity:</strong> {sentiment_data['polarity']}<br>
                <strong>Subjectivity:</strong> {sentiment_data['subjectivity']}
            </div>
            """, unsafe_allow_html=True)
        
        if 'readability' in results:
            readability = results['readability']
            st.markdown(f"""
            <div class="metric-card">
                <strong>Reading Level:</strong> {readability['difficulty_level']}<br>
                <strong>Flesch Score:</strong> {readability['flesch_reading_ease']}<br>
                <strong>Avg Sentence Length:</strong> {readability['avg_sentence_length']}
            </div>
            """, unsafe_allow_html=True)
    
    # Keywords visualization
    if 'keywords' in results:
        st.markdown("### 🔍 Keywords Analysis")
        
        keywords_df = pd.DataFrame(results['keywords'], columns=['Keyword', 'Score'])
        keywords_df = keywords_df.head(15)
        
        fig_keywords = px.bar(
            keywords_df,
            x='Score',
            y='Keyword',
            title="Top Keywords (TF-IDF Scores)",
            orientation='h'
        )
        fig_keywords.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_keywords, use_container_width=True)
    
    # Word cloud data
    if 'word_cloud_data' in results:
        st.markdown("### ☁️ Word Frequency")
        
        word_freq = results['word_cloud_data']['word_frequencies']
        if word_freq:
            # Create word frequency chart
            words = list(word_freq.keys())[:20]
            frequencies = list(word_freq.values())[:20]
            
            fig_wordcloud = go.Figure(data=[go.Bar(
                x=frequencies,
                y=words,
                orientation='h',
                marker_color='lightblue'
            )])
            fig_wordcloud.update_layout(
                title="Top 20 Most Frequent Words",
                xaxis_title="Frequency",
                yaxis_title="Words",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_wordcloud, use_container_width=True)
    
    # Named entities
    if 'entities' in results:
        st.markdown("### 👥 Named Entities")
        
        entities = results['entities']
        entity_data = []
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_data.append({'Type': entity_type, 'Entity': entity})
        
        if entity_data:
            entity_df = pd.DataFrame(entity_data)
            
            # Count entities by type
            entity_counts = entity_df['Type'].value_counts()
            
            fig_entities = px.pie(
                values=entity_counts.values,
                names=entity_counts.index,
                title="Named Entities Distribution"
            )
            st.plotly_chart(fig_entities, use_container_width=True)
            
            # Show entities table
            st.dataframe(entity_df, use_container_width=True)
    
    # GenAI insights
    if 'genai_insights' in results and 'insights' in results['genai_insights']:
        st.markdown("### 🧠 AI-Generated Insights")
        
        insights = results['genai_insights']['insights']
        for i, insight in enumerate(insights, 1):
            st.markdown(f"""
            <div class="insight-box">
                <strong>Insight {i}:</strong> {insight}
            </div>
            """, unsafe_allow_html=True)

def qa_assistant_tab():
    """Q&A assistant tab"""
    if not st.session_state.current_analysis:
        st.info("👆 Please upload and analyze a document first")
        return
    
    st.markdown('<h2 class="sub-header">💬 Q&A Assistant</h2>', unsafe_allow_html=True)
    
    results = st.session_state.current_analysis
    text = results['text']
    
    st.markdown(f"**Document:** {results['filename']}")
    st.markdown("Ask questions about your document and get AI-powered answers!")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?",
        help="Ask specific questions about the document content"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        answer_method = st.selectbox(
            "Answer Method:",
            ["auto", "openai", "local"],
            help="Choose the AI model to use for answering"
        )
    
    if question and st.button("🤔 Get Answer", type="primary"):
        with st.spinner("Thinking..."):
            try:
                answer_result = genai_analyzer.answer_question(text, question, method=answer_method)
                
                # Display answer
                st.markdown("### 💡 Answer")
                
                if 'error' not in answer_result:
                    st.success(answer_result['answer'])
                    
                    # Show confidence if available
                    if 'confidence' in answer_result:
                        confidence = answer_result['confidence']
                        st.progress(confidence)
                        st.caption(f"Confidence: {confidence:.2%} | Method: {answer_result['method']}")
                else:
                    st.error(f"Error: {answer_result['error']}")
                
            except Exception as e:
                st.error(f"Failed to generate answer: {str(e)}")
    
    # Suggested questions
    st.markdown("### 💭 Suggested Questions")
    
    if results.get('classification', {}).get('document_type'):
        doc_type = results['classification']['document_type']
        
        suggestions = {
            'Technical/Academic': [
                "What is the main research question?",
                "What methodology was used?",
                "What are the key findings?"
            ],
            'Business/Financial': [
                "What are the key financial metrics?",
                "What is the business strategy?",
                "What are the market opportunities?"
            ],
            'Legal/Compliance': [
                "What are the main legal requirements?",
                "What are the compliance obligations?",
                "What are the penalties mentioned?"
            ]
        }
        
        if doc_type in suggestions:
            for suggestion in suggestions[doc_type]:
                if st.button(f"💬 {suggestion}", key=f"suggest_{suggestion}"):
                    st.experimental_set_query_params(question=suggestion)

def document_library_tab():
    """Document library tab"""
    st.markdown('<h2 class="sub-header">📚 Document Library</h2>', unsafe_allow_html=True)
    
    if not st.session_state.analyzed_documents:
        st.info("No documents analyzed yet. Upload and analyze documents to see them here.")
        return
    
    st.markdown(f"**Total Documents:** {len(st.session_state.analyzed_documents)}")
    
    # Document list
    for i, doc in enumerate(st.session_state.analyzed_documents):
        with st.expander(f"📄 {doc['filename']} - {doc['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Words", f"{doc['document_stats']['word_count']:,}")
                if 'sentiment' in doc:
                    st.write(f"**Sentiment:** {doc['sentiment']['sentiment']}")
            
            with col2:
                if 'classification' in doc:
                    st.write(f"**Type:** {doc['classification']['document_type']}")
                if 'readability' in doc:
                    st.write(f"**Reading Level:** {doc['readability']['difficulty_level']}")
            
            with col3:
                if st.button(f"Load for Analysis", key=f"load_{i}"):
                    st.session_state.current_analysis = doc
                    st.success("Document loaded!")
                    st.experimental_rerun()
            
            # Document preview
            if st.button(f"Show Preview", key=f"preview_{i}"):
                st.text_area("Document Preview", doc['text'][:500] + "...", height=100)
    
    # Clear library
    if st.button("🗑️ Clear Library", type="secondary"):
        st.session_state.analyzed_documents = []
        st.session_state.current_analysis = None
        st.success("Library cleared!")
        st.experimental_rerun()

if __name__ == "__main__":
    main()
