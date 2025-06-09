"""
Enhanced Software Engineering Word Embeddings
Implementation of the proposed enhancement using transformers and modern NLP tools
"""

import os
import json
import pickle
import requests
import wikipedia
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap

# Text processing
import spacy
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class WikipediaDataCollector:
    """Enhanced data collection using Wikipedia API"""
    
    def __init__(self):
        self.se_categories = [
            "Software engineering",
            "Software development",
            "Programming languages",
            "Software testing",
            "Software architecture",
            "Requirements engineering",
            "Software design patterns"
        ]
        
    def collect_se_pages(self, max_pages_per_category: int = 50) -> List[str]:
        """Collect software engineering pages using Wikipedia API"""
        all_texts = []
        
        for category in self.se_categories:
            try:
                # Search for pages in this category
                search_results = wikipedia.search(category, results=max_pages_per_category)
                
                for title in search_results[:max_pages_per_category]:
                    try:
                        page = wikipedia.page(title)
                        # Filter for SE-related content
                        if self._is_se_related(page.content):
                            all_texts.append(page.content)
                            print(f"Collected: {title}")
                    except (wikipedia.exceptions.DisambiguationError, 
                           wikipedia.exceptions.PageError) as e:
                        continue
                        
            except Exception as e:
                print(f"Error collecting from category {category}: {e}")
                continue
                
        return all_texts
    
    def _is_se_related(self, text: str) -> bool:
        """Check if text is software engineering related"""
        se_keywords = [
            'software', 'programming', 'development', 'algorithm', 'code',
            'testing', 'debugging', 'architecture', 'design pattern',
            'requirements', 'engineering', 'computer science'
        ]
        text_lower = text.lower()
        return sum(keyword in text_lower for keyword in se_keywords) >= 3

class EnhancedPreprocessor:
    """Advanced preprocessing using spaCy and transformer tokenizers"""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Advanced preprocessing pipeline"""
        processed_texts = []
        
        for text in texts:
            # Basic cleaning
            text = self._clean_text(text)
            
            # spaCy processing
            doc = self.nlp(text)
            
            # Extract meaningful tokens
            tokens = []
            for token in doc:
                if self._is_valid_token(token):
                    tokens.append(token.lemma_.lower())
            
            if len(tokens) > 50:  # Minimum length threshold
                processed_texts.append(' '.join(tokens))
                
        return processed_texts
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Preserve technical terms with special characters
        # This is important for SE terminology like C++, .NET, etc.
        return text
    
    def _is_valid_token(self, token) -> bool:
        """Check if token should be kept"""
        return (
            not token.is_stop and
            not token.is_punct and
            not token.is_space and
            len(token.text) > 2 and
            token.text.lower() not in self.stop_words and
            token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']
        )

class TransformerEmbeddings:
    """Contextual embeddings using transformer models"""
    
    def __init__(self, model_name: str = 'microsoft/codebert-base'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def get_embeddings(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Generate contextual embeddings for texts"""
        embeddings = []
        
        for text in texts:
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
        
        for word1, word2 in word_pairs:
            emb1 = self.get_embeddings([word1])
            emb2 = self.get_embeddings([word2])
            sim = cosine_similarity(emb1, emb2)[0][0]
            similarities.append(sim)
            
        return similarities

class OriginalWord2VecModel:
    """Reproduction of the original Word2Vec approach"""
    
    def __init__(self):
        self.model = None
        self.vocabulary = set()
        
    def train(self, texts: List[str], vector_size: int = 50, window: int = 10, 
              min_count: int = 1, epochs: int = 5):
        """Train Word2Vec model similar to original paper"""
        # Tokenize texts
        sentences = []
        for text in texts:
            tokens = text.split()
            # Preserve special characters for SE terms
            sentences.append(tokens)
            self.vocabulary.update(tokens)
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            sg=1  # Skip-gram
        )
        
        print(f"Trained Word2Vec model with vocabulary size: {len(self.model.wv.key_to_index)}")
        
    def get_similar_words(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """Get most similar words"""
        try:
            return self.model.wv.most_similar(word, topn=topn)
        except KeyError:
            return []
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector"""
        try:
            return self.model.wv[word]
        except KeyError:
            return None

class EmbeddingAnalyzer:
    """Analysis and visualization tools for embeddings"""
    
    def __init__(self):
        pass
    
    def visualize_embeddings(self, embeddings: np.ndarray, labels: List[str], 
                           method: str = 'umap', title: str = 'Embedding Visualization'):
        """Visualize embeddings using dimensionality reduction"""
        
        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42)
            
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Create interactive plot
        fig = px.scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            text=labels,
            title=f'{title} ({method.upper()})',
            labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'}
        )
        
        fig.update_traces(textposition="top center")
        return fig
    
    def compare_similarities(self, word_pairs: List[Tuple[str, str]], 
                           transformer_sims: List[float], 
                           word2vec_sims: List[float]) -> go.Figure:
        """Compare similarities between different models"""
        
        fig = go.Figure()
        
        # Add transformer similarities
        fig.add_trace(go.Scatter(
            x=list(range(len(word_pairs))),
            y=transformer_sims,
            mode='lines+markers',
            name='Transformer (CodeBERT)',
            line=dict(color='blue')
        ))
        
        # Add Word2Vec similarities
        fig.add_trace(go.Scatter(
            x=list(range(len(word_pairs))),
            y=word2vec_sims,
            mode='lines+markers',
            name='Word2Vec (Original)',
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title='Semantic Similarity Comparison',
            xaxis_title='Word Pairs',
            yaxis_title='Cosine Similarity',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(word_pairs))),
                ticktext=[f"{pair[0]}-{pair[1]}" for pair in word_pairs]
            )
        )
        
        return fig

def main():
    """Main implementation demonstrating the enhanced approach"""
    
    print("=== Enhanced SE Word Embeddings Implementation ===")
    
    # 1. Data Collection
    print("\n1. Collecting Software Engineering data...")
    collector = WikipediaDataCollector()
    raw_texts = collector.collect_se_pages(max_pages_per_category=20)
    print(f"Collected {len(raw_texts)} SE-related documents")
    
    # 2. Preprocessing
    print("\n2. Advanced preprocessing with spaCy...")
    preprocessor = EnhancedPreprocessor()
    processed_texts = preprocessor.preprocess_texts(raw_texts)
    print(f"Processed {len(processed_texts)} documents")
    
    # 3. Enhanced Model Training
    print("\n3. Training enhanced transformer model...")
    transformer_model = TransformerEmbeddings('microsoft/codebert-base')
    
    # Sample texts for demonstration
    sample_texts = processed_texts[:10] if len(processed_texts) >= 10 else processed_texts
    transformer_embeddings = transformer_model.get_embeddings(sample_texts)
    
    # 4. Original Model for Comparison
    print("\n4. Training original Word2Vec model...")
    original_model = OriginalWord2VecModel()
    original_model.train(processed_texts)
    
    # 5. Comparative Analysis
    print("\n5. Conducting comparative analysis...")
    
    # Test word pairs for similarity comparison
    test_pairs = [
        ('software', 'program'),
        ('bug', 'error'),
        ('class', 'object'),
        ('algorithm', 'method'),
        ('testing', 'debugging')
    ]
    
    # Get similarities from both models
    transformer_sims = transformer_model.get_word_similarities(test_pairs)
    
    word2vec_sims = []
    for word1, word2 in test_pairs:
        vec1 = original_model.get_word_vector(word1)
        vec2 = original_model.get_word_vector(word2)
        if vec1 is not None and vec2 is not None:
            sim = cosine_similarity([vec1], [vec2])[0][0]
            word2vec_sims.append(sim)
        else:
            word2vec_sims.append(0.0)
    
    # 6. Visualization
    print("\n6. Creating visualizations...")
    analyzer = EmbeddingAnalyzer()
    
    # Similarity comparison
    comparison_fig = analyzer.compare_similarities(test_pairs, transformer_sims, word2vec_sims)
    comparison_fig.write_html('/home/ubuntu/se_embeddings_enhanced/similarity_comparison.html')
    
    # Embedding visualization
    sample_labels = [f"Doc_{i}" for i in range(len(sample_texts))]
    embedding_fig = analyzer.visualize_embeddings(
        transformer_embeddings, 
        sample_labels, 
        method='umap',
        title='Enhanced SE Embeddings (CodeBERT)'
    )
    embedding_fig.write_html('/home/ubuntu/se_embeddings_enhanced/embedding_visualization.html')
    
    # 7. Results Summary
    print("\n=== RESULTS SUMMARY ===")
    print(f"Documents processed: {len(processed_texts)}")
    print(f"Original Word2Vec vocabulary: {len(original_model.vocabulary)}")
    print(f"Transformer model: {transformer_model.model_name}")
    
    print("\nSimilarity Comparison:")
    for i, (word1, word2) in enumerate(test_pairs):
        print(f"{word1}-{word2}: Transformer={transformer_sims[i]:.3f}, Word2Vec={word2vec_sims[i]:.3f}")
    
    print("\nVisualization files saved:")
    print("- similarity_comparison.html")
    print("- embedding_visualization.html")
    
    # Save models and results
    results = {
        'test_pairs': test_pairs,
        'transformer_similarities': transformer_sims,
        'word2vec_similarities': word2vec_sims,
        'num_documents': len(processed_texts),
        'vocabulary_size': len(original_model.vocabulary)
    }
    
    with open('/home/ubuntu/se_embeddings_enhanced/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nImplementation completed successfully!")
    return results

if __name__ == "__main__":
    results = main()

