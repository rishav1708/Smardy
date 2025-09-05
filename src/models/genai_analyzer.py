"""
GenAI Analysis Module
Integrates with LLM APIs for advanced document analysis, summarization, and Q&A
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Union
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
    GenAI-powered document analyzer using Hugging Face and local models:
    - Text summarization (extractive and abstractive)
    - Question answering
    - Content generation
    - Semantic similarity
    - Advanced insights extraction
    """
    
    def __init__(self, use_huggingface: bool = True):
        self.use_huggingface = use_huggingface
        self.hf_api_key = os.getenv('HUGGING_FACE_API_KEY')
        self.hf_api_url = "https://api-inference.huggingface.co/models/"
        
        # Hugging Face is available even without API key (rate limited)
        self.huggingface_available = True
        if self.hf_api_key:
            logger.info("✅ Hugging Face API key found")
        else:
            logger.info("ℹ️ Using Hugging Face free tier (rate limited)")
        
        # Initialize local models
        self._init_local_models()
    
    def _query_huggingface(self, model_name: str, payload: dict, max_retries: int = 3) -> dict:
        """
        Query Hugging Face Inference API with retries
        """
        headers = {}
        if self.hf_api_key:
            headers["Authorization"] = f"Bearer {self.hf_api_key}"
        
        url = f"{self.hf_api_url}{model_name}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Model is loading, wait a bit
                    import time
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    logger.warning(f"HF API error {response.status_code}: {response.text}")
                    return {"error": f"API error {response.status_code}"}
            except Exception as e:
                logger.warning(f"HF API attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
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
            method = "huggingface" if self.huggingface_available else "local"
        
        try:
            if method == "huggingface" and self.huggingface_available:
                return self._summarize_with_huggingface(text, max_length)
            else:
                return self._summarize_with_local_model(text, max_length, min_length)
                
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return {
                'summary': self._extractive_summary(text, 3),
                'method': 'fallback',
                'error': str(e)
            }
    
    def _summarize_with_huggingface(self, text: str, max_length: int) -> Dict:
        """Generate summary using Hugging Face models"""
        try:
            # Truncate text if too long
            max_input_length = 1000  # Keep within model limits
            words = text.split()
            if len(words) > max_input_length:
                text = ' '.join(words[:max_input_length])
            
            # Try different summarization models in order of preference
            models_to_try = [
                "facebook/bart-large-cnn",
                "microsoft/DialoGPT-medium",
                "t5-small"
            ]
            
            for model_name in models_to_try:
                try:
                    payload = {
                        "inputs": text,
                        "parameters": {
                            "max_length": max_length,
                            "min_length": max(10, max_length // 4),
                            "do_sample": False
                        }
                    }
                    
                    result = self._query_huggingface(model_name, payload)
                    
                    if "error" not in result and result:
                        if isinstance(result, list) and len(result) > 0:
                            summary = result[0].get('summary_text', '')
                            if summary:
                                return {
                                    'summary': summary,
                                    'method': f'huggingface_{model_name.split("/")[-1]}',
                                    'word_count': len(summary.split()),
                                    'compression_ratio': round(len(text.split()) / len(summary.split()), 2)
                                }
                except Exception as e:
                    logger.warning(f"Failed with model {model_name}: {str(e)}")
                    continue
            
            # If all models fail, raise exception to trigger fallback
            raise ValueError("All Hugging Face summarization models failed")
            
        except Exception as e:
            logger.error(f"Hugging Face summarization error: {str(e)}")
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
        Answer questions about the document with robust fallback handling
        """
        if method == "auto":
            # Try Hugging Face first, then local models, then simple search
            method = "huggingface" if self.huggingface_available else "local"
        
        try:
            if method == "huggingface" and self.huggingface_available:
                return self._qa_with_huggingface(text, question)
            else:
                return self._qa_with_local_model(text, question)
                
        except Exception as e:
            logger.error(f"Question answering error: {str(e)}")
            # If everything fails, try simple search as last resort
            try:
                return self._qa_with_simple_search(text, question)
            except Exception as e2:
                logger.error(f"All Q&A methods failed: {str(e2)}")
                return {
                    'answer': "I'm having trouble analyzing your question. Please try asking about specific topics mentioned in the document.",
                    'confidence': 0.0,
                    'method': 'error_fallback',
                    'error': str(e)
                }
    
    def _qa_with_huggingface(self, text: str, question: str) -> Dict:
        """Answer question using Hugging Face models"""
        try:
            # Truncate text if too long
            max_context_length = 512
            words = text.split()
            if len(words) > max_context_length:
                text = ' '.join(words[:max_context_length])
            
            # Try different Q&A models
            models_to_try = [
                "deepset/roberta-base-squad2",
                "distilbert-base-cased-distilled-squad",
                "microsoft/DialoGPT-medium"
            ]
            
            for model_name in models_to_try:
                try:
                    payload = {
                        "inputs": {
                            "question": question,
                            "context": text
                        }
                    }
                    
                    result = self._query_huggingface(model_name, payload)
                    
                    if "error" not in result and result:
                        if isinstance(result, dict):
                            answer = result.get('answer', '')
                            score = result.get('score', 0.0)
                            
                            if answer and len(answer.strip()) > 0:
                                return {
                                    'answer': answer,
                                    'confidence': round(float(score), 3),
                                    'method': f'huggingface_{model_name.split("/")[-1]}',
                                    'start_position': result.get('start', -1),
                                    'end_position': result.get('end', -1)
                                }
                        
                except Exception as e:
                    logger.warning(f"HF Q&A model {model_name} failed: {str(e)}")
                    continue
            
            # If all models fail, raise to trigger local fallback
            raise ValueError("All Hugging Face Q&A models failed")
            
        except Exception as e:
            logger.error(f"Hugging Face Q&A error: {str(e)}")
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
        Generate advanced insights about the document using Hugging Face
        """
        try:
            if self.huggingface_available:
                return self._generate_insights_with_huggingface(text)
            else:
                # Use local insights as fallback
                insights = self._generate_local_insights(text)
                return {
                    'insights': insights,
                    'method': 'local_analysis'
                }
        except Exception as e:
            logger.error(f"Insights generation error: {str(e)}")
            # Fallback to local insights
            try:
                insights = self._generate_local_insights(text)
                return {
                    'insights': insights,
                    'method': 'local_fallback'
                }
            except Exception as e2:
                return {
                    'insights': ["Unable to generate insights at this time"],
                    'error': str(e)
                }
    
    def _generate_insights_with_huggingface(self, text: str) -> Dict:
        """
        Generate insights using Hugging Face text generation models
        """
        # Truncate text if too long
        max_tokens = 500
        words = text.split()
        if len(words) > max_tokens:
            text = ' '.join(words[:max_tokens])
        
        # Try different text generation models for insights
        models_to_try = [
            "microsoft/DialoGPT-medium",
            "gpt2",
            "distilgpt2"
        ]
        
        prompt = f"Analyze this document and provide key insights: {text}\n\nKey insights:"
        
        for model_name in models_to_try:
            try:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": 200,
                        "temperature": 0.7,
                        "do_sample": True
                    }
                }
                
                result = self._query_huggingface(model_name, payload)
                
                if "error" not in result and result:
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        # Extract insights from generated text
                        insights_part = generated_text.replace(prompt, '').strip()
                        insights = [insight.strip() for insight in insights_part.split('.') if insight.strip()]
                        
                        if insights and len(insights) > 0:
                            return {
                                'insights': insights[:5],  # Limit to 5 insights
                                'method': f'huggingface_{model_name.split("/")[-1]}'
                            }
            except Exception as e:
                logger.warning(f"HF insights model {model_name} failed: {str(e)}")
                continue
        
        # If all HF models fail, use local insights
        insights = self._generate_local_insights(text)
        return {
            'insights': insights,
            'method': 'local_fallback'
        }
    
    def _generate_local_insights(self, text: str) -> List[str]:
        """
        Generate basic insights using local text analysis
        """
        words = text.split()
        sentences = text.split('. ')
        
        insights = []
        
        # Document length insight
        word_count = len(words)
        if word_count < 100:
            insights.append("This appears to be a short document or excerpt")
        elif word_count > 1000:
            insights.append("This is a comprehensive document with substantial content")
        else:
            insights.append("This document contains a moderate amount of content")
        
        # Sentence structure insight
        avg_sentence_length = word_count / max(len(sentences), 1)
        if avg_sentence_length > 20:
            insights.append("The document uses complex sentence structures")
        elif avg_sentence_length < 10:
            insights.append("The document uses simple, direct language")
        else:
            insights.append("The document has balanced sentence complexity")
        
        # Content analysis based on common patterns
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['research', 'study', 'analysis', 'methodology']):
            insights.append("This appears to be research or academic content")
        
        if any(word in text_lower for word in ['business', 'market', 'revenue', 'profit']):
            insights.append("This document contains business-related information")
        
        if any(word in text_lower for word in ['legal', 'contract', 'agreement', 'terms']):
            insights.append("This appears to contain legal or contractual content")
        
        # If no specific insights were found, add a general one
        if len(insights) == 2:  # Only length and sentence structure insights
            insights.append("The document contains varied content suitable for analysis")
        
        return insights
    
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
        Generate keywords using Hugging Face for contextual understanding
        """
        try:
            if self.huggingface_available:
                # Try to use HF text generation for keyword extraction
                prompt = f"Extract the most important keywords from this text: {text[:800]}\n\nKeywords:"
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": 50,
                        "temperature": 0.1,
                        "do_sample": True
                    }
                }
                
                result = self._query_huggingface("gpt2", payload)
                
                if "error" not in result and result:
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        keywords_part = generated_text.replace(prompt, '').strip()
                        keywords = [kw.strip() for kw in keywords_part.split(',') if kw.strip()]
                        
                        if keywords:
                            return {
                                'keywords': keywords[:num_keywords],
                                'method': 'huggingface_gpt2'
                            }
            
            # Fallback to simple keyword extraction
            return self._extract_keywords_local(text, num_keywords)
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}")
            return self._extract_keywords_local(text, num_keywords)
    
    def _extract_keywords_local(self, text: str, num_keywords: int) -> Dict:
        """
        Extract keywords using local text analysis methods
        """
        try:
            import re
            from collections import Counter
            
            # Clean and tokenize text
            words = re.findall(r'\b\w{3,}\b', text.lower())
            
            # Common stop words to filter out
            stop_words = {'the', 'and', 'are', 'for', 'with', 'this', 'that', 'from', 'they', 'have', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'more', 'very', 'into', 'after', 'first', 'well', 'also', 'where', 'much', 'than', 'only', 'its', 'now', 'way', 'may', 'when', 'them', 'some', 'what', 'make', 'like', 'him', 'her', 'how', 'did', 'get', 'has', 'had', 'who'}
            
            # Filter out stop words and count word frequency
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            word_counts = Counter(filtered_words)
            
            # Get most common words
            keywords = [word for word, _ in word_counts.most_common(num_keywords)]
            
            return {
                'keywords': keywords,
                'method': 'local_frequency'
            }
        except Exception as e:
            return {
                'keywords': [],
                'error': str(e),
                'method': 'error'
            }
    
    def check_api_status(self) -> Dict:
        """
        Check the status of Hugging Face API and local models
        """
        status = {
            'huggingface_available': self.huggingface_available,
            'local_models': {
                'summarizer': self.summarizer is not None,
                'qa_model': self.qa_model is not None,
                'sentence_model': self.sentence_model is not None
            }
        }
        
        if self.huggingface_available:
            try:
                # Test Hugging Face connection with a simple request
                test_result = self._query_huggingface(
                    "gpt2", 
                    {"inputs": "test", "parameters": {"max_length": 10}}
                )
                if "error" not in test_result:
                    status['huggingface_connection'] = 'working'
                else:
                    status['huggingface_connection'] = f'limited: {test_result.get("error", "unknown")}'
            except Exception as e:
                status['huggingface_connection'] = f'error: {str(e)}'
        
        return status
