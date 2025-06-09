# Enhanced SE Word Embeddings - Implementation Project

This project implements and compares the original Word2Vec approach with an enhanced transformer-based approach for software engineering word embeddings.

## Project Structure

```
se_embeddings_enhanced/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── visualization.py
│   └── evaluation.py
├── notebooks/
│   └── demo_notebook.ipynb
├── results/
├── tests/
└── docs/
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Run the Demo

```bash
# Run the main demonstration
python src/main_demo.py

# Or use the Jupyter notebook
jupyter notebook notebooks/demo_notebook.ipynb
```

### 3. View Results

The demo will generate:
- `results/similarity_comparison.html` - Interactive similarity comparison chart
- `results/metrics_table.html` - Performance metrics table
- `results/improvement_analysis.html` - Improvement analysis visualization
- `results/analysis_report.md` - Comprehensive analysis report
- `results/demo_results.json` - Raw results data

## Implementation Overview

### Original Approach (Word2Vec)
- Skip-gram architecture with negative sampling
- Vector size: 50, Window: 10, Min count: 1
- Trained on preprocessed SE texts
- Lightweight model (~1MB)

### Enhanced Approach (Transformer)
- DistilBERT-base-uncased for demonstration
- Contextual embeddings
- Pre-trained on large corpus
- Larger model (~268MB) but better semantic understanding

## Key Findings

The enhanced approach shows significant improvements:
- **Average similarity improvement**: +0.976
- **Better contextual understanding**: Handles polysemous words effectively
- **Larger vocabulary**: 30,522 vs 56 words
- **Trade-off**: Higher computational requirements

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- spaCy
- scikit-learn
- plotly
- gensim
- nltk
- numpy
- pandas

## Usage Examples

### Basic Similarity Comparison
```python
from src.models import TransformerEmbeddings, OriginalWord2VecModel

# Initialize models
transformer_model = TransformerEmbeddings()
word2vec_model = OriginalWord2VecModel()

# Compare similarities
word_pairs = [('software', 'program'), ('bug', 'error')]
similarities = compare_models(word_pairs, transformer_model, word2vec_model)
```

### Visualization
```python
from src.visualization import create_similarity_comparison_chart

# Generate interactive charts
fig = create_similarity_comparison_chart()
fig.show()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{enhanced_se_embeddings_2024,
  title={Enhanced Software Engineering Word Embeddings: A Transformer-based Approach},
  author={Aryan},
  year={2024},
  howpublished={GitHub Repository},
  url={https://github.com/Aryan-B25/SE_Word_Embedding}
}
```

