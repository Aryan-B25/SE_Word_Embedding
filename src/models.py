"""
Models Module for SE Word Embeddings
"""

import os
import json
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TransformerEmbeddings:
    """Contextual embeddings using transformer models"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name
        print(f"Loading transformer model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Model loaded on device: {self.device}")
        
    def get_embeddings(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Generate contextual embeddings for texts"""
        embeddings = []
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Processing text {i+1}/{len(texts)}")
                
            # Tokenize and encode
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                max_length=max_length, 
                truncation=True, 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding as sentence representation
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
                
        return np.array(embeddings)
    
    def get_word_similarities(self, word_pairs: List[Tuple[str, str]]) -> List[float]:
        """Calculate semantic similarities between word pairs"""
        similarities = []
        
        print(f"Calculating similarities for {len(word_pairs)} word pairs...")
        
        for word1, word2 in word_pairs:
            emb1 = self.get_embeddings([word1])
            emb2 = self.get_embeddings([word2])
            sim = cosine_similarity(emb1, emb2)[0][0]
            similarities.append(float(sim))
            
        return similarities
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a single word"""
        return self.get_embeddings([word])[0]

class OriginalWord2VecModel:
    """Reproduction of the original Word2Vec approach"""
    
    def __init__(self):
        self.model = None
        self.vocabulary = set()
        
    def train(self, texts: List[str], vector_size: int = 50, window: int = 10, 
              min_count: int = 1, epochs: int = 5, workers: int = 4):
        """Train Word2Vec model similar to original paper"""
        
        print(f"Training Word2Vec model on {len(texts)} texts...")
        
        # Tokenize texts
        sentences = []
        for text in texts:
            tokens = text.split()
            sentences.append(tokens)
            self.vocabulary.update(tokens)
        
        print(f"Total vocabulary before training: {len(self.vocabulary)} words")
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers,
            sg=1  # Skip-gram
        )
        
        print(f"Word2Vec training complete. Final vocabulary: {len(self.model.wv.key_to_index)} words")
        
    def get_similar_words(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """Get most similar words"""
        try:
            return self.model.wv.most_similar(word, topn=topn)
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
            return []
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector"""
        try:
            return self.model.wv[word]
        except KeyError:
            return None
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words"""
        try:
            return self.model.wv.similarity(word1, word2)
        except KeyError:
            return 0.0
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.model = Word2Vec.load(filepath)
            self.vocabulary = set(self.model.wv.key_to_index.keys())
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file not found: {filepath}")
            return False

class ModelComparator:
    """Compare different embedding models"""
    
    def __init__(self, transformer_model: TransformerEmbeddings, 
                 word2vec_model: OriginalWord2VecModel):
        self.transformer_model = transformer_model
        self.word2vec_model = word2vec_model
    
    def compare_similarities(self, word_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Compare similarities between models"""
        results = []
        
        print(f"Comparing similarities for {len(word_pairs)} word pairs...")
        
        for word1, word2 in word_pairs:
            # Word2Vec similarity
            w2v_sim = self.word2vec_model.calculate_similarity(word1, word2)
            
            # Transformer similarity
            trans_sim = self.transformer_model.get_word_similarities([(word1, word2)])[0]
            
            result = {
                'word_pair': f"{word1}-{word2}",
                'word1': word1,
                'word2': word2,
                'word2vec_similarity': float(w2v_sim),
                'transformer_similarity': float(trans_sim),
                'improvement': float(trans_sim - w2v_sim)
            }
            
            results.append(result)
            print(f"  {word1}-{word2}: W2V={w2v_sim:.3f}, Trans={trans_sim:.3f}, Diff={trans_sim-w2v_sim:+.3f}")
            
        return results
    
    def evaluate_vocabulary_coverage(self, test_words: List[str]) -> Dict:
        """Evaluate vocabulary coverage"""
        w2v_coverage = sum(1 for word in test_words if word in self.word2vec_model.model.wv.key_to_index)
        trans_coverage = len(test_words)  # Transformer models handle all words via subword tokenization
        
        return {
            'test_words': len(test_words),
            'word2vec_coverage': w2v_coverage,
            'transformer_coverage': trans_coverage,
            'word2vec_coverage_rate': w2v_coverage / len(test_words),
            'transformer_coverage_rate': 1.0
        }
    
    def get_model_statistics(self) -> Dict:
        """Get statistics about both models"""
        return {
            'word2vec': {
                'vocabulary_size': len(self.word2vec_model.model.wv.key_to_index),
                'vector_size': self.word2vec_model.model.wv.vector_size,
                'model_size_mb': 1  # Approximate
            },
            'transformer': {
                'model_name': self.transformer_model.model_name,
                'vocabulary_size': self.transformer_model.tokenizer.vocab_size,
                'hidden_size': self.transformer_model.model.config.hidden_size,
                'model_size_mb': 268  # Approximate for DistilBERT
            }
        }

