"""
GenAI Analysis Module
Integrates with LLM APIs for advanced document analysis, summarization, and Q&A
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available")
# from transformers import pipeline, AutoTokenizer, AutoModel  # Local model loading disabled for Python 3.13
# import torch
# from sentence_transformers import SentenceTransformer  # Not available in Python 3.13
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to get API key from Streamlit secrets if available (only in cloud deployment)
try:
    import streamlit as st
    # Only try to access secrets if we're in a Streamlit app context and secrets exist
    if hasattr(st, 'secrets') and st.secrets is not None:
        try:
            if 'OPENAI_API_KEY' in st.secrets:
                os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
        except Exception:
            # Secrets not available, continue with environment variables
            pass
except (ImportError, KeyError, AttributeError):
    # Streamlit not available or secrets not configured
    pass

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
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        # Initialize OpenAI if API key is available
        if self.use_openai and self.openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.openai_available = True
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                self.openai_available = False
                self.openai_client = None
        else:
            self.openai_available = False
            self.openai_client = None
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI package not available")
            elif not self.openai_api_key:
                logger.warning("OpenAI API key not found. Checking for alternatives...")
        
        # Initialize Hugging Face as fallback (FREE)
        self.huggingface_available = False
        if self.huggingface_api_key:
            self.huggingface_available = True
            logger.info("Hugging Face API available as free alternative")
        else:
            logger.info("Hugging Face API key not found - using local methods")
        
        # Initialize local models
        self._init_local_models()
    
    def _init_local_models(self):
        """Initialize local transformer models (disabled for Python 3.13 compatibility)"""
        # All local transformer models disabled for Python 3.13 compatibility
        self.summarizer = None
        self.qa_model = None
        self.sentence_model = None
        logger.info("Local models disabled for Python 3.13 compatibility")
    
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
        if not self.openai_client:
            raise ValueError("OpenAI client not available")
            
        try:
            # Truncate text if too long (GPT-3.5 has token limits)
            max_tokens = 3000  # Leave room for prompt and response
            words = text.split()
            if len(words) > max_tokens:
                text = ' '.join(words[:max_tokens])
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional document summarizer. Create concise, informative summaries that capture the key points and main ideas."
                    },
                    {
                        "role": "user", 
                        "content": f"Please summarize the following text in approximately {max_length} words, focusing on the main points and key insights:\n\n{text}"
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
        """Generate summary using local BART model"""
        if not self.summarizer:
            raise ValueError("Local summarization model not available")
        
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
            elif method == "huggingface" and self.huggingface_available:
                return self._qa_with_huggingface(text, question)
            elif method == "auto":
                if self.openai_available:
                    return self._qa_with_openai(text, question)
                elif self.huggingface_available:
                    return self._qa_with_huggingface(text, question)
                else:
                    return self._qa_with_local_model(text, question)
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
        if not self.openai_client:
            raise ValueError("OpenAI client not available")
            
        try:
            # Truncate text if too long
            max_tokens = 2500
            words = text.split()
            if len(words) > max_tokens:
                text = ' '.join(words[:max_tokens])
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided documents. If you cannot find the answer in the document, say so clearly."
                    },
                    {
                        "role": "user",
                        "content": f"Based on the following document, please answer this question: {question}\n\nDocument:\n{text}"
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
        """Answer question using local model or simple text search"""
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
                logger.error(f"Local QA error: {str(e)}")
                # Fall through to simple search
        
        # Simple keyword-based answer extraction as fallback
        return self._simple_qa_fallback(text, question)
    
    def _qa_with_huggingface(self, text: str, question: str) -> Dict:
        """Answer question using Hugging Face free API"""
        try:
            import requests
            
            # Use free Hugging Face inference API
            API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
            headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
            
            # Truncate text if too long
            max_length = 500
            words = text.split()
            if len(words) > max_length:
                text = ' '.join(words[:max_length])
            
            payload = {
                "inputs": {
                    "question": question,
                    "context": text
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if 'answer' in result:
                    return {
                        'answer': result['answer'],
                        'confidence': round(result.get('score', 0.5), 3),
                        'method': 'huggingface_free'
                    }
            
            # If API fails, fall back to local method
            return self._simple_qa_fallback(text, question)
            
        except Exception as e:
            logger.error(f"Hugging Face QA error: {str(e)}")
            return self._simple_qa_fallback(text, question)
    
    def _simple_qa_fallback(self, text: str, question: str) -> Dict:
        """Simple keyword-based Q&A fallback"""
        try:
            question_lower = question.lower()
            text_lower = text.lower()
            sentences = text.split('. ')
            
            # Extract key question words
            question_words = [word for word in question_lower.split() 
                            if word not in ['what', 'where', 'when', 'who', 'how', 'why', 'is', 'are', 'the', 'a', 'an']]
            
            # Find sentences containing question keywords
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for word in question_words if word in sentence_lower)
                if score > 0:
                    relevant_sentences.append((score, sentence.strip()))
            
            if relevant_sentences:
                # Sort by relevance and take the best match
                relevant_sentences.sort(key=lambda x: x[0], reverse=True)
                best_sentence = relevant_sentences[0][1]
                
                # If the sentence is too long, try to extract the most relevant part
                if len(best_sentence.split()) > 30:
                    words = best_sentence.split()
                    for word in question_words:
                        if word in best_sentence.lower():
                            word_idx = best_sentence.lower().find(word)
                            start_idx = max(0, word_idx - 50)
                            end_idx = min(len(best_sentence), word_idx + 100)
                            best_sentence = best_sentence[start_idx:end_idx].strip()
                            break
                
                return {
                    'answer': best_sentence,
                    'confidence': min(0.7, relevant_sentences[0][0] * 0.2),
                    'method': 'simple_keyword_search'
                }
            else:
                return {
                    'answer': "I couldn't find specific information to answer that question in the document.",
                    'confidence': 0.0,
                    'method': 'simple_keyword_search'
                }
                
        except Exception as e:
            logger.error(f"Simple QA fallback error: {str(e)}")
            return {
                'answer': "I'm unable to process that question at the moment.",
                'confidence': 0.0,
                'method': 'error',
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
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document analyst. Provide insightful analysis including key themes, potential implications, and notable patterns in the text."
                    },
                    {
                        "role": "user",
                        "content": f"Please analyze the following document and provide key insights, themes, and observations:\n\n{text}"
                    }
                ],
                max_tokens=300,
                temperature=0.4
            )
            
            insights_text = response.choices[0].message.content.strip()
            
            # Split insights into list
            insights = [insight.strip() for insight in insights_text.split('\n') if insight.strip()]
            
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
            response = self.openai_client.chat.completions.create(
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
        
        if self.openai_available and self.openai_client:
            try:
                # Test OpenAI connection with a minimal request
                self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1
                )
                status['openai_connection'] = 'working'
            except Exception as e:
                status['openai_connection'] = f'error: {str(e)}'
                status['openai_available'] = False
        
        return status
