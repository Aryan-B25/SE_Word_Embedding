"""
Preprocessing Module for SE Word Embeddings
"""

import re
import spacy
import nltk
from nltk.corpus import stopwords
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EnhancedPreprocessor:
    """Advanced preprocessing using spaCy and transformer tokenizers"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        self.stop_words = set(stopwords.words('english'))
        
        # SE-specific terms to preserve
        self.se_terms = {
            'c++', 'c#', '.net', 'api', 'sql', 'html', 'css', 'javascript',
            'python', 'java', 'php', 'ruby', 'go', 'rust', 'swift',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask',
            'git', 'github', 'docker', 'kubernetes', 'aws', 'azure'
        }
        
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Advanced preprocessing pipeline"""
        if isinstance(texts[0], dict):
            # Extract content from dictionary format
            text_contents = [item['content'] for item in texts]
        else:
            text_contents = texts
            
        processed_texts = []
        
        print(f"Preprocessing {len(text_contents)} texts...")
        
        for i, text in enumerate(text_contents):
            if i % 10 == 0:
                print(f"  Processing text {i+1}/{len(text_contents)}")
                
            # Basic cleaning
            text = self._clean_text(text)
            
            if self.nlp:
                # spaCy processing
                processed_text = self._spacy_process(text)
            else:
                # Fallback to simple processing
                processed_text = self._simple_process(text)
            
            if len(processed_text.split()) > 20:  # Minimum length threshold
                processed_texts.append(processed_text)
                
        print(f"Preprocessing complete. {len(processed_texts)} texts retained.")
        return processed_texts
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve SE terms
        text = re.sub(r'[^\w\s\.\+\#\-]', ' ', text)
        
        return text.strip()
    
    def _spacy_process(self, text: str) -> str:
        """Process text using spaCy"""
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            if self._is_valid_token(token):
                # Preserve original form for SE terms, otherwise use lemma
                if token.text.lower() in self.se_terms:
                    tokens.append(token.text.lower())
                else:
                    tokens.append(token.lemma_.lower())
        
        return ' '.join(tokens)
    
    def _simple_process(self, text: str) -> str:
        """Simple processing fallback"""
        # Basic tokenization and cleaning
        words = text.lower().split()
        words = [word.strip('.,!?;:"()[]') for word in words]
        words = [word for word in words if self._is_valid_simple_token(word)]
        
        return ' '.join(words)
    
    def _is_valid_token(self, token) -> bool:
        """Check if spaCy token should be kept"""
        return (
            not token.is_stop and
            not token.is_punct and
            not token.is_space and
            len(token.text) > 2 and
            token.text.lower() not in self.stop_words and
            (token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN'] or 
             token.text.lower() in self.se_terms)
        )
    
    def _is_valid_simple_token(self, word: str) -> bool:
        """Check if simple token should be kept"""
        return (
            len(word) > 2 and
            word not in self.stop_words and
            word.isalpha()
        )

class SimplePreprocessor:
    """Simple preprocessing for quick testing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Simple preprocessing pipeline"""
        if isinstance(texts[0], dict):
            text_contents = [item['content'] for item in texts]
        else:
            text_contents = texts
            
        processed = []
        
        for text in text_contents:
            # Simple tokenization and cleaning
            words = text.lower().split()
            words = [word.strip('.,!?;:"()[]') for word in words]
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            if len(words) > 20:
                processed.append(' '.join(words))
                
        return processed

