"""
Traditional ML Analysis Module
Implements document classification, sentiment analysis, and clustering using scikit-learn
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# NLP Libraries  
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from textblob import TextBlob
# from wordcloud import WordCloud  # Removed due to Python 3.13 compatibility

# Download required NLTK data with SSL bypass
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def safe_nltk_download(resource):
    """Safely download NLTK data with error handling"""
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        try:
            nltk.download(resource.split('/')[-1], quiet=True)
            return True
        except Exception as e:
            logger.warning(f"Could not download NLTK resource {resource}: {str(e)}")
            return False

# Try to download required NLTK data
safe_nltk_download('tokenizers/punkt')
safe_nltk_download('corpora/stopwords')
safe_nltk_download('corpora/wordnet')
safe_nltk_download('taggers/averaged_perceptron_tagger')
safe_nltk_download('chunkers/maxent_ne_chunker')
safe_nltk_download('corpora/words')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe tokenization functions with fallbacks
def safe_word_tokenize(text):
    """Safely tokenize text with fallback"""
    try:
        return word_tokenize(text)
    except Exception:
        # Simple fallback tokenization
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

def safe_sent_tokenize(text):
    """Safely tokenize sentences with fallback"""
    try:
        return sent_tokenize(text)
    except Exception:
        # Simple fallback sentence tokenization
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class MLDocumentAnalyzer:
    """
    Traditional ML-based document analyzer with multiple capabilities:
    - Text classification
    - Sentiment analysis  
    - Topic modeling
    - Document clustering
    - Keyword extraction
    - Named entity recognition
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.classifier = MultinomialNB()
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        self.topic_model = LatentDirichletAllocation(n_components=5, random_state=42)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to get NLTK stopwords, fallback to basic English stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Could not load NLTK stopwords: {str(e)}")
            # Fallback basic English stopwords
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        # Document categories for classification
        self.document_categories = {
            0: "Technical/Academic",
            1: "Business/Financial", 
            2: "Legal/Compliance",
            3: "Marketing/Sales",
            4: "News/Media",
            5: "Personal/Informal"
        }
    
    def preprocess_for_ml(self, text: str, use_stemming: bool = False) -> str:
        """Preprocess text for ML analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = safe_word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Stemming or Lemmatization
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob and rule-based approach
        """
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment_label = "Positive"
        elif polarity < -0.1:
            sentiment_label = "Negative" 
        else:
            sentiment_label = "Neutral"
            
        # Confidence score
        confidence = abs(polarity)
        
        return {
            'sentiment': sentiment_label,
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'confidence': round(confidence, 3)
        }
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF
        """
        processed_text = self.preprocess_for_ml(text)
        
        # Create a temporary vectorizer for keyword extraction
        keyword_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        try:
            tfidf_matrix = keyword_vectorizer.fit_transform([processed_text])
            feature_names = keyword_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:num_keywords]
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}")
            return []
    
    def extract_named_entities(self, text: str) -> Dict:
        """
        Extract named entities using NLTK
        """
        try:
            tokens = safe_word_tokenize(text)
            pos_tags = pos_tag(tokens)
            entities = ne_chunk(pos_tags, binary=False)
            
            entity_dict = {
                'PERSON': [],
                'ORGANIZATION': [],
                'GPE': [],  # Geopolitical entity
                'OTHER': []
            }
            
            current_entity = ""
            current_label = ""
            
            for chunk in entities:
                if hasattr(chunk, 'label'):
                    if current_entity and current_label:
                        if current_label in entity_dict:
                            entity_dict[current_label].append(current_entity.strip())
                        else:
                            entity_dict['OTHER'].append(current_entity.strip())
                    
                    current_entity = " ".join([token for token, pos in chunk])
                    current_label = chunk.label()
                else:
                    if current_entity and current_label:
                        if current_label in entity_dict:
                            entity_dict[current_label].append(current_entity.strip())
                        else:
                            entity_dict['OTHER'].append(current_entity.strip())
                        current_entity = ""
                        current_label = ""
            
            # Remove duplicates
            for key in entity_dict:
                entity_dict[key] = list(set(entity_dict[key]))
            
            return entity_dict
            
        except Exception as e:
            logger.error(f"Named entity extraction error: {str(e)}")
            return {'PERSON': [], 'ORGANIZATION': [], 'GPE': [], 'OTHER': []}
    
    def classify_document_type(self, text: str) -> Dict:
        """
        Classify document type using keyword-based rules
        """
        text_lower = text.lower()
        
        # Define keyword patterns for different document types
        patterns = {
            'Technical/Academic': [
                'research', 'study', 'analysis', 'methodology', 'algorithm', 
                'implementation', 'experiment', 'results', 'conclusion', 'abstract'
            ],
            'Business/Financial': [
                'revenue', 'profit', 'investment', 'financial', 'budget', 
                'market', 'business', 'strategy', 'growth', 'performance'
            ],
            'Legal/Compliance': [
                'contract', 'agreement', 'terms', 'conditions', 'legal', 
                'compliance', 'regulation', 'policy', 'law', 'clause'
            ],
            'Marketing/Sales': [
                'marketing', 'sales', 'customer', 'product', 'brand', 
                'campaign', 'advertising', 'promotion', 'target', 'market'
            ],
            'News/Media': [
                'reported', 'according', 'sources', 'breaking', 'news', 
                'journalist', 'media', 'press', 'statement', 'announced'
            ]
        }
        
        scores = {}
        for doc_type, keywords in patterns.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            scores[doc_type] = score
        
        # Get the document type with highest score
        predicted_type = max(scores, key=scores.get) if max(scores.values()) > 0 else "General"
        confidence = scores[predicted_type] / len(text.split()) * 100 if predicted_type != "General" else 0
        
        return {
            'document_type': predicted_type,
            'confidence': round(confidence, 2),
            'scores': scores
        }
    
    def perform_topic_modeling(self, texts: List[str], num_topics: int = 5) -> Dict:
        """
        Perform topic modeling using LDA
        """
        if len(texts) < 2:
            return {'topics': [], 'error': 'Need at least 2 documents for topic modeling'}
        
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_for_ml(text) for text in texts]
            
            # Vectorize
            vectorizer = CountVectorizer(max_features=100, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            
            # LDA
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': [round(topic[i], 4) for i in top_words_idx]
                })
            
            return {'topics': topics}
            
        except Exception as e:
            logger.error(f"Topic modeling error: {str(e)}")
            return {'topics': [], 'error': str(e)}
    
    def cluster_documents(self, texts: List[str], num_clusters: int = 3) -> Dict:
        """
        Cluster documents using K-means
        """
        if len(texts) < num_clusters:
            return {'clusters': [], 'error': f'Need at least {num_clusters} documents for clustering'}
        
        try:
            # Preprocess and vectorize
            processed_texts = [self.preprocess_for_ml(text) for text in texts]
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Organize results
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'document_id': i,
                    'preview': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i]
                })
            
            return {
                'clusters': [{'cluster_id': k, 'documents': v} for k, v in clusters.items()],
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }
            
        except Exception as e:
            logger.error(f"Document clustering error: {str(e)}")
            return {'clusters': [], 'error': str(e)}
    
    def generate_word_cloud_data(self, text: str) -> Dict:
        """
        Generate word frequency data for word cloud
        """
        try:
            processed_text = self.preprocess_for_ml(text)
            words = processed_text.split()
            
            # Count word frequencies
            word_freq = Counter(words)
            
            # Remove very common words that might not be informative
            common_words = {'would', 'could', 'should', 'one', 'two', 'also', 'get', 'go', 'see'}
            word_freq = {word: count for word, count in word_freq.items() 
                        if word not in common_words and len(word) > 2}
            
            # Get top 50 words
            top_words = dict(word_freq.most_common(50))
            
            return {
                'word_frequencies': top_words,
                'total_unique_words': len(word_freq),
                'most_common_word': max(word_freq, key=word_freq.get) if word_freq else None
            }
            
        except Exception as e:
            logger.error(f"Word cloud generation error: {str(e)}")
            return {'word_frequencies': {}, 'error': str(e)}
    
    def analyze_readability(self, text: str) -> Dict:
        """
        Analyze text readability using various metrics
        """
        sentences = safe_sent_tokenize(text)
        words = safe_word_tokenize(text)
        words = [word for word in words if word.isalpha()]
        
        if not sentences or not words:
            return {'error': 'Text too short for readability analysis'}
        
        # Basic metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Syllable counting (simplified)
        def count_syllables(word):
            vowels = 'aeiouy'
            word = word.lower()
            syllables = 0
            prev_char = ''
            for char in word:
                if char in vowels and prev_char not in vowels:
                    syllables += 1
                prev_char = char
            return max(1, syllables)
        
        total_syllables = sum(count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Flesch Reading Ease Score (simplified)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100
        
        # Difficulty level
        if flesch_score >= 90:
            difficulty = "Very Easy"
        elif flesch_score >= 80:
            difficulty = "Easy"
        elif flesch_score >= 70:
            difficulty = "Fairly Easy"
        elif flesch_score >= 60:
            difficulty = "Standard"
        elif flesch_score >= 50:
            difficulty = "Fairly Difficult"
        elif flesch_score >= 30:
            difficulty = "Difficult"
        else:
            difficulty = "Very Difficult"
        
        return {
            'flesch_reading_ease': round(flesch_score, 2),
            'difficulty_level': difficulty,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2),
            'total_sentences': len(sentences),
            'total_words': len(words),
            'total_syllables': total_syllables
        }
