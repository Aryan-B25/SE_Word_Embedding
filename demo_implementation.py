"""
Simplified Enhanced SE Word Embeddings Implementation
Optimized version for demonstration purposes
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Text processing
import spacy
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SimplifiedImplementation:
    """Simplified implementation for demonstration"""
    
    def __init__(self):
        print("Initializing simplified SE embeddings implementation...")
        
        # Sample SE texts for demonstration
        self.sample_texts = [
            "Software engineering is the systematic application of engineering approaches to software development.",
            "Object-oriented programming uses classes and objects to structure code effectively.",
            "Unit testing involves testing individual components of software in isolation.",
            "Agile development emphasizes iterative development and customer collaboration.",
            "Design patterns provide reusable solutions to common programming problems.",
            "Version control systems track changes in source code during development.",
            "Code review is a systematic examination of computer source code.",
            "Software architecture defines the fundamental structures of a software system.",
            "Requirements engineering involves eliciting, analyzing, and documenting software requirements.",
            "Debugging is the process of finding and resolving defects in computer programs."
        ]
        
        # SE-specific vocabulary for testing
        self.se_vocabulary = [
            'software', 'engineering', 'programming', 'development', 'testing',
            'debugging', 'algorithm', 'architecture', 'design', 'pattern',
            'class', 'object', 'method', 'function', 'variable',
            'bug', 'error', 'exception', 'interface', 'implementation'
        ]
        
    def preprocess_texts(self, texts):
        """Simple preprocessing"""
        processed = []
        stop_words = set(stopwords.words('english'))
        
        for text in texts:
            # Simple tokenization and cleaning
            words = text.lower().split()
            words = [word.strip('.,!?;:"()[]') for word in words]
            words = [word for word in words if word not in stop_words and len(word) > 2]
            processed.append(' '.join(words))
            
        return processed
    
    def train_original_word2vec(self, texts):
        """Train Word2Vec model similar to original paper"""
        sentences = [text.split() for text in texts]
        
        model = Word2Vec(
            sentences=sentences,
            vector_size=50,
            window=10,
            min_count=1,
            epochs=5,
            sg=1  # Skip-gram
        )
        
        return model
    
    def get_transformer_embeddings(self, texts, model_name='distilbert-base-uncased'):
        """Get transformer embeddings (using smaller model for demo)"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
                
        return np.array(embeddings)
    
    def compare_similarities(self, word_pairs, word2vec_model, transformer_tokenizer, transformer_model):
        """Compare similarities between models"""
        results = []
        
        for word1, word2 in word_pairs:
            # Word2Vec similarity
            try:
                w2v_sim = word2vec_model.wv.similarity(word1, word2)
            except KeyError:
                w2v_sim = 0.0
            
            # Transformer similarity
            inputs1 = transformer_tokenizer(word1, return_tensors='pt')
            inputs2 = transformer_tokenizer(word2, return_tensors='pt')
            
            with torch.no_grad():
                emb1 = transformer_model(**inputs1).last_hidden_state[:, 0, :].numpy()
                emb2 = transformer_model(**inputs2).last_hidden_state[:, 0, :].numpy()
                trans_sim = cosine_similarity(emb1, emb2)[0][0]
            
            results.append({
                'word_pair': f"{word1}-{word2}",
                'word2vec_similarity': float(w2v_sim),
                'transformer_similarity': float(trans_sim)
            })
            
        return results
    
    def run_demonstration(self):
        """Run the complete demonstration"""
        print("\n=== SE Word Embeddings Comparison Demo ===")
        
        # 1. Preprocess texts
        print("\n1. Preprocessing texts...")
        processed_texts = self.preprocess_texts(self.sample_texts)
        print(f"Processed {len(processed_texts)} documents")
        
        # 2. Train original Word2Vec
        print("\n2. Training Word2Vec model (original approach)...")
        word2vec_model = self.train_original_word2vec(processed_texts)
        print(f"Word2Vec vocabulary size: {len(word2vec_model.wv.key_to_index)}")
        
        # 3. Load transformer model
        print("\n3. Loading transformer model (enhanced approach)...")
        transformer_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        transformer_model = AutoModel.from_pretrained('distilbert-base-uncased')
        print("Transformer model loaded successfully")
        
        # 4. Compare similarities
        print("\n4. Comparing semantic similarities...")
        test_pairs = [
            ('software', 'program'),
            ('bug', 'error'),
            ('class', 'object'),
            ('testing', 'debugging'),
            ('design', 'architecture')
        ]
        
        similarity_results = self.compare_similarities(
            test_pairs, word2vec_model, transformer_tokenizer, transformer_model
        )
        
        # 5. Display results
        print("\n=== COMPARISON RESULTS ===")
        print(f"{'Word Pair':<20} {'Word2Vec':<12} {'Transformer':<12} {'Difference':<12}")
        print("-" * 60)
        
        for result in similarity_results:
            diff = result['transformer_similarity'] - result['word2vec_similarity']
            print(f"{result['word_pair']:<20} {result['word2vec_similarity']:<12.3f} "
                  f"{result['transformer_similarity']:<12.3f} {diff:<12.3f}")
        
        # 6. Test word similarities
        print("\n=== WORD SIMILARITY EXAMPLES ===")
        test_words = ['software', 'programming', 'testing', 'design']
        
        for word in test_words:
            if word in word2vec_model.wv.key_to_index:
                similar_words = word2vec_model.wv.most_similar(word, topn=3)
                print(f"\nWord2Vec - Most similar to '{word}':")
                for sim_word, score in similar_words:
                    print(f"  {sim_word}: {score:.3f}")
        
        # 7. Save results
        results_data = {
            'similarity_comparisons': similarity_results,
            'vocabulary_size': len(word2vec_model.wv.key_to_index),
            'num_documents': len(processed_texts),
            'test_pairs': test_pairs
        }
        
        with open('/home/ubuntu/se_embeddings_enhanced/demo_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n=== SUMMARY ===")
        print(f"Documents processed: {len(processed_texts)}")
        print(f"Word2Vec vocabulary: {len(word2vec_model.wv.key_to_index)} words")
        print(f"Transformer model: distilbert-base-uncased")
        print(f"Results saved to: demo_results.json")
        
        return results_data

def main():
    """Main demonstration function"""
    demo = SimplifiedImplementation()
    results = demo.run_demonstration()
    return results

if __name__ == "__main__":
    results = main()

