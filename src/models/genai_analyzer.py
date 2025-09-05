"""
GenAI Analysis Module
Integrates with LLM APIs for advanced document analysis, summarization, and Q&A
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
import openai
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenAIDocumentAnalyzer:
    """
    GenAI-powered document analyzer using modern LLM capabilities:
    - Text summarization (extractive and abstractive)
    - Question answering
    - Content generation
    - Semantic similarity
    - Advanced insights extraction
    """
    
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize OpenAI if API key is available
        if self.use_openai and self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.openai_available = True
        else:
            self.openai_available = False
            logger.warning("OpenAI API key not found. Using local models only.")
        
        # Initialize local models
        self._init_local_models()
    
    def _init_local_models(self):
        """Initialize local transformer models with graceful fallbacks"""
        # Initialize models as None first
        self.summarizer = None
        self.qa_model = None
        self.sentence_model = None
        
        try:
            # Try to initialize summarization model
            logger.info("Attempting to load summarization model...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn"
            )
            logger.info("✅ Summarization model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load summarization model: {str(e)}")
            self.summarizer = None
        
        try:
            # Try to initialize Q&A model with a lighter alternative first
            logger.info("Attempting to load Q&A model...")
            # Use a smaller, faster model for Streamlit Cloud
            self.qa_model = pipeline(
                "question-answering",
                model="distilbert-base-uncased-distilled-squad"
            )
            logger.info("✅ Q&A model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Q&A model: {str(e)}")
            try:
                # Try an even lighter model
                logger.info("Trying lighter Q&A model...")
                self.qa_model = pipeline(
                    "question-answering",
                    model="deepset/minilm-uncased-squad2"
                )
                logger.info("✅ Light Q&A model loaded successfully")
            except Exception as e2:
                logger.error(f"❌ All Q&A models failed to load: {str(e2)}")
                self.qa_model = None
        
        try:
            # Try to initialize sentence transformer
            logger.info("Attempting to load sentence transformer...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load sentence transformer: {str(e)}")
            self.sentence_model = None
        
        # Log final status
        models_loaded = sum([self.summarizer is not None, 
                           self.qa_model is not None, 
                           self.sentence_model is not None])
        logger.info(f"Local model initialization complete: {models_loaded}/3 models loaded successfully")
    
    def generate_summary(self, text: str, method: str = "auto", 
                        max_length: int = 150, min_length: int = 50) -> Dict:
        """
        Generate document summary using various methods
        
        Args:
            text: Input text to summarize
            method: "openai", "local", or "auto"
            max_length: Maximum length of summary
            min_length: Minimum length of summary
        """
        if len(text.split()) < 20:
            return {
                'summary': text,
                'method': 'none',
                'error': 'Text too short for summarization'
            }
        
        # Choose method
        if method == "auto":
            method = "openai" if self.openai_available else "local"
        
        try:
            if method == "openai" and self.openai_available:
                return self._summarize_with_openai(text, max_length)
            else:
                return self._summarize_with_local_model(text, max_length, min_length)
                
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return {
                'summary': self._extractive_summary(text, 3),
                'method': 'fallback',
                'error': str(e)
            }
    
    def _summarize_with_openai(self, text: str, max_length: int) -> Dict:
        """Generate summary using OpenAI GPT"""
        try:
            # Truncate text if too long (GPT-3.5 has token limits)
            max_tokens = 3000  # Leave room for prompt and response
            words = text.split()
            if len(words) > max_tokens:
                text = ' '.join(words[:max_tokens])
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional document summarizer. Create concise, informative summaries that capture the key points and main ideas."
                    },
                    {
                        "role": "user", 
                        "content": f"Please summarize the following text in approximately {max_length} words, focusing on the main points and key insights:\\n\\n{text}"
                    }
                ],
                max_tokens=max_length * 2,  # Allow some buffer
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                'summary': summary,
                'method': 'openai',
                'word_count': len(summary.split()),
                'compression_ratio': round(len(text.split()) / len(summary.split()), 2)
            }
            
        except Exception as e:
            logger.error(f"OpenAI summarization error: {str(e)}")
            raise e
    
    def _summarize_with_local_model(self, text: str, max_length: int, min_length: int) -> Dict:
        """Generate summary using local BART model with fallback"""
        if not self.summarizer:
            # Use extractive summary as fallback
            logger.info("Using extractive summary fallback")
            return {
                'summary': self._extractive_summary(text, 3),
                'method': 'extractive_fallback',
                'word_count': len(self._extractive_summary(text, 3).split()),
                'compression_ratio': round(len(text.split()) / len(self._extractive_summary(text, 3).split()), 2)
            }
        
        try:
            # Split text into chunks if too long
            max_chunk_length = 1024  # BART max length
            words = text.split()
            
            if len(words) <= max_chunk_length:
                chunks = [text]
            else:
                chunk_size = max_chunk_length - 50  # Leave buffer
                chunks = []
                for i in range(0, len(words), chunk_size):
                    chunks.append(' '.join(words[i:i + chunk_size]))
            
            summaries = []
            for chunk in chunks:
                summary_result = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summaries.append(summary_result[0]['summary_text'])
            
            # Combine summaries if multiple chunks
            if len(summaries) > 1:
                combined_summary = ' '.join(summaries)
                # Summarize the combined summary if it's still too long
                if len(combined_summary.split()) > max_length:
                    final_summary = self.summarizer(
                        combined_summary,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )[0]['summary_text']
                else:
                    final_summary = combined_summary
            else:
                final_summary = summaries[0]
            
            return {
                'summary': final_summary,
                'method': 'local_bart',
                'word_count': len(final_summary.split()),
                'compression_ratio': round(len(text.split()) / len(final_summary.split()), 2)
            }
            
        except Exception as e:
            logger.error(f"Local summarization error: {str(e)}")
            raise e
    
    def _extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """Generate extractive summary as fallback"""
        try:
            sentences = text.split('. ')
            if len(sentences) <= num_sentences:
                return text
            
            # Simple scoring based on sentence length and position
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = len(sentence.split())  # Word count
                if i < len(sentences) * 0.3:  # Boost early sentences
                    score *= 1.2
                scored_sentences.append((score, sentence))
            
            # Get top sentences
            top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
            selected_sentences = [sent for _, sent in top_sentences[:num_sentences]]
            
            return '. '.join(selected_sentences) + '.'
            
        except Exception:
            return text[:500] + '...' if len(text) > 500 else text
    
    def answer_question(self, text: str, question: str, method: str = "auto") -> Dict:
        """
        Answer questions about the document
        """
        if method == "auto":
            method = "openai" if self.openai_available else "local"
        
        try:
            if method == "openai" and self.openai_available:
                return self._qa_with_openai(text, question)
            else:
                return self._qa_with_local_model(text, question)
                
        except Exception as e:
            logger.error(f"Question answering error: {str(e)}")
            return {
                'answer': "I'm unable to answer that question at the moment.",
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def _qa_with_openai(self, text: str, question: str) -> Dict:
        """Answer question using OpenAI"""
        try:
            # Truncate text if too long
            max_tokens = 2500
            words = text.split()
            if len(words) > max_tokens:
                text = ' '.join(words[:max_tokens])
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided documents. If you cannot find the answer in the document, say so clearly."
                    },
                    {
                        "role": "user",
                        "content": f"Based on the following document, please answer this question: {question}\\n\\nDocument:\\n{text}"
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'method': 'openai',
                'confidence': 0.85  # OpenAI doesn't provide confidence scores
            }
            
        except Exception as e:
            logger.error(f"OpenAI QA error: {str(e)}")
            raise e
    
    def _qa_with_local_model(self, text: str, question: str) -> Dict:
        """Answer question using local model with fallback"""
        if self.qa_model:
            try:
                # Truncate text if too long
                max_length = 512  # DistilBERT limit
                words = text.split()
                if len(words) > max_length:
                    text = ' '.join(words[:max_length])
                
                result = self.qa_model(question=question, context=text)
                
                return {
                    'answer': result['answer'],
                    'confidence': round(result['score'], 3),
                    'method': 'local_distilbert',
                    'start_position': result.get('start', -1),
                    'end_position': result.get('end', -1)
                }
                
            except Exception as e:
                logger.error(f"Local QA model error: {str(e)}")
                return self._qa_with_simple_search(text, question)
        else:
            # Use simple fallback when no model is available
            return self._qa_with_simple_search(text, question)
    
    def _qa_with_simple_search(self, text: str, question: str) -> Dict:
        """
        Simple keyword-based search fallback for Q&A when models are unavailable
        """
        try:
            import re
            from collections import Counter
            
            # Extract key terms from the question
            question_words = re.findall(r'\b\w{3,}\b', question.lower())
            # Remove common question words
            stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'the', 'and', 'are', 'for', 'with', 'this', 'that'}
            key_terms = [word for word in question_words if word not in stop_words]
            
            if not key_terms:
                return {
                    'answer': "I need more specific keywords in your question to search the document.",
                    'confidence': 0.1,
                    'method': 'simple_search_fallback'
                }
            
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            sentence_scores = []
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                    
                sentence_lower = sentence.lower()
                score = 0
                
                # Score based on keyword matches
                for term in key_terms:
                    if term in sentence_lower:
                        score += sentence_lower.count(term)
                
                if score > 0:
                    sentence_scores.append((score, sentence.strip()))
            
            if sentence_scores:
                # Sort by score and return the best match
                sentence_scores.sort(key=lambda x: x[0], reverse=True)
                best_sentence = sentence_scores[0][1]
                confidence = min(sentence_scores[0][0] / len(key_terms) / 2, 0.8)  # Normalize confidence
                
                return {
                    'answer': best_sentence,
                    'confidence': round(confidence, 3),
                    'method': 'simple_search_fallback'
                }
            else:
                return {
                    'answer': "I couldn't find relevant information in the document to answer your question. Try asking about specific topics mentioned in the document.",
                    'confidence': 0.0,
                    'method': 'simple_search_fallback'
                }
                
        except Exception as e:
            logger.error(f"Simple search fallback error: {str(e)}")
            return {
                'answer': "I'm unable to process your question at the moment. Please try rephrasing it.",
                'confidence': 0.0,
                'method': 'fallback_error',
                'error': str(e)
            }
    
    def generate_insights(self, text: str) -> Dict:
        """
        Generate advanced insights about the document using GenAI
        """
        if not self.openai_available:
            return {
                'insights': ["GenAI insights require OpenAI API key"],
                'error': 'OpenAI API not available'
            }
        
        try:
            # Truncate text if too long
            max_tokens = 2000
            words = text.split()
            if len(words) > max_tokens:
                text = ' '.join(words[:max_tokens])
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document analyst. Provide insightful analysis including key themes, potential implications, and notable patterns in the text."
                    },
                    {
                        "role": "user",
                        "content": f"Please analyze the following document and provide key insights, themes, and observations:\\n\\n{text}"
                    }
                ],
                max_tokens=300,
                temperature=0.4
            )
            
            insights_text = response.choices[0].message.content.strip()
            
            # Split insights into list
            insights = [insight.strip() for insight in insights_text.split('\\n') if insight.strip()]
            
            return {
                'insights': insights,
                'method': 'openai'
            }
            
        except Exception as e:
            logger.error(f"Insights generation error: {str(e)}")
            return {
                'insights': ["Error generating insights"],
                'error': str(e)
            }
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> Dict:
        """
        Compute semantic similarity between two texts using sentence embeddings
        """
        if not self.sentence_model:
            return {'similarity': 0.0, 'error': 'Sentence transformer not available'}
        
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode([text1, text2])
            
            # Compute cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            similarity_percentage = similarity * 100
            
            # Interpret similarity
            if similarity_percentage >= 80:
                interpretation = "Very High"
            elif similarity_percentage >= 60:
                interpretation = "High"
            elif similarity_percentage >= 40:
                interpretation = "Moderate"
            elif similarity_percentage >= 20:
                interpretation = "Low"
            else:
                interpretation = "Very Low"
            
            return {
                'similarity_score': round(float(similarity), 4),
                'similarity_percentage': round(similarity_percentage, 2),
                'interpretation': interpretation
            }
            
        except Exception as e:
            logger.error(f"Similarity computation error: {str(e)}")
            return {'similarity': 0.0, 'error': str(e)}
    
    def generate_keywords_genai(self, text: str, num_keywords: int = 10) -> Dict:
        """
        Generate keywords using GenAI for more contextual understanding
        """
        if not self.openai_available:
            return {'keywords': [], 'error': 'OpenAI API not available'}
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"Extract the {num_keywords} most important keywords or key phrases from the text. Return them as a comma-separated list."
                    },
                    {
                        "role": "user",
                        "content": text[:2000]  # Limit text length
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            
            return {
                'keywords': keywords[:num_keywords],
                'method': 'genai'
            }
            
        except Exception as e:
            logger.error(f"GenAI keyword extraction error: {str(e)}")
            return {'keywords': [], 'error': str(e)}
    
    def check_api_status(self) -> Dict:
        """
        Check the status of various API connections and models
        """
        status = {
            'openai_available': self.openai_available,
            'local_models': {
                'summarizer': self.summarizer is not None,
                'qa_model': self.qa_model is not None,
                'sentence_model': self.sentence_model is not None
            }
        }
        
        if self.openai_available:
            try:
                # Test OpenAI connection with a minimal request
                openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1
                )
                status['openai_connection'] = 'working'
            except Exception as e:
                status['openai_connection'] = f'error: {str(e)}'
                status['openai_available'] = False
        
        return status
