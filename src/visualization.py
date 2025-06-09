"""
Visualization Module for SE Word Embeddings
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
import os

class EmbeddingVisualizer:
    """Visualization tools for embedding analysis"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def create_similarity_comparison_chart(self, comparison_results: List[Dict]) -> go.Figure:
        """Create comparison chart for similarities"""
        
        word_pairs = [item['word_pair'] for item in comparison_results]
        word2vec_sims = [item['word2vec_similarity'] for item in comparison_results]
        transformer_sims = [item['transformer_similarity'] for item in comparison_results]
        
        fig = go.Figure()
        
        # Add Word2Vec similarities
        fig.add_trace(go.Scatter(
            x=word_pairs,
            y=word2vec_sims,
            mode='lines+markers',
            name='Word2Vec (Original)',
            line=dict(color='red', width=3),
            marker=dict(size=10, symbol='circle')
        ))
        
        # Add Transformer similarities
        fig.add_trace(go.Scatter(
            x=word_pairs,
            y=transformer_sims,
            mode='lines+markers',
            name='Transformer (Enhanced)',
            line=dict(color='blue', width=3),
            marker=dict(size=10, symbol='diamond')
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Semantic Similarity Comparison: Original vs Enhanced Approach',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title='Word Pairs',
            yaxis_title='Cosine Similarity',
            font=dict(size=12),
            legend=dict(x=0.02, y=0.98),
            width=900,
            height=600,
            template='plotly_white'
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        # Save chart
        filepath = os.path.join(self.results_dir, 'similarity_comparison.html')
        fig.write_html(filepath)
        print(f"Similarity comparison chart saved to {filepath}")
        
        return fig
    
    def create_improvement_analysis(self, comparison_results: List[Dict]) -> go.Figure:
        """Create improvement analysis chart"""
        
        word_pairs = [item['word_pair'] for item in comparison_results]
        improvements = [item['improvement'] for item in comparison_results]
        
        # Color bars based on improvement (green for positive, red for negative)
        colors = ['green' if x > 0 else 'red' for x in improvements]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=word_pairs,
            y=improvements,
            name='Improvement',
            marker_color=colors,
            text=[f"{x:+.3f}" for x in improvements],
            textposition='outside'
        ))
        
        fig.update_layout(
            title={
                'text': 'Semantic Similarity Improvement: Transformer vs Word2Vec',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title='Word Pairs',
            yaxis_title='Improvement (Transformer - Word2Vec)',
            width=900,
            height=600,
            template='plotly_white'
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)
        
        # Save chart
        filepath = os.path.join(self.results_dir, 'improvement_analysis.html')
        fig.write_html(filepath)
        print(f"Improvement analysis chart saved to {filepath}")
        
        return fig
    
    def create_performance_metrics_table(self, comparison_results: List[Dict], 
                                       model_stats: Dict) -> go.Figure:
        """Create performance metrics comparison table"""
        
        # Calculate metrics
        word2vec_sims = [item['word2vec_similarity'] for item in comparison_results]
        transformer_sims = [item['transformer_similarity'] for item in comparison_results]
        improvements = [item['improvement'] for item in comparison_results]
        
        metrics_data = {
            'Metric': [
                'Average Similarity',
                'Max Similarity', 
                'Min Similarity',
                'Std Deviation',
                'Average Improvement',
                'Vocabulary Size',
                'Model Size (MB)',
                'Vector Dimensions'
            ],
            'Word2Vec (Original)': [
                f"{np.mean(word2vec_sims):.3f}",
                f"{np.max(word2vec_sims):.3f}",
                f"{np.min(word2vec_sims):.3f}",
                f"{np.std(word2vec_sims):.3f}",
                "0.000",
                f"{model_stats['word2vec']['vocabulary_size']:,}",
                f"{model_stats['word2vec']['model_size_mb']}",
                f"{model_stats['word2vec']['vector_size']}"
            ],
            'Transformer (Enhanced)': [
                f"{np.mean(transformer_sims):.3f}",
                f"{np.max(transformer_sims):.3f}",
                f"{np.min(transformer_sims):.3f}",
                f"{np.std(transformer_sims):.3f}",
                f"{np.mean(improvements):+.3f}",
                f"{model_stats['transformer']['vocabulary_size']:,}",
                f"{model_stats['transformer']['model_size_mb']}",
                f"{model_stats['transformer']['hidden_size']}"
            ],
            'Difference': [
                f"{np.mean(transformer_sims) - np.mean(word2vec_sims):+.3f}",
                f"{np.max(transformer_sims) - np.max(word2vec_sims):+.3f}",
                f"{np.min(transformer_sims) - np.min(word2vec_sims):+.3f}",
                f"{np.std(transformer_sims) - np.std(word2vec_sims):+.3f}",
                f"{np.mean(improvements):+.3f}",
                f"{model_stats['transformer']['vocabulary_size'] / model_stats['word2vec']['vocabulary_size']:.1f}x",
                f"{model_stats['transformer']['model_size_mb'] / model_stats['word2vec']['model_size_mb']:.0f}x",
                f"{model_stats['transformer']['hidden_size'] / model_stats['word2vec']['vector_size']:.1f}x"
            ]
        }
        
        df = pd.DataFrame(metrics_data)
        
        # Create table visualization
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='lightblue',
                align='center',
                font=dict(size=14, color='black'),
                height=40
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color=[['white', 'lightgray'] * len(df)],
                align='center',
                font=dict(size=12),
                height=35
            )
        )])
        
        fig.update_layout(
            title={
                'text': 'Performance Metrics Comparison',
                'x': 0.5,
                'font': {'size': 16}
            },
            width=1000,
            height=500
        )
        
        # Save table
        filepath = os.path.join(self.results_dir, 'metrics_table.html')
        fig.write_html(filepath)
        print(f"Performance metrics table saved to {filepath}")
        
        return fig
    
    def create_vocabulary_coverage_chart(self, coverage_data: Dict) -> go.Figure:
        """Create vocabulary coverage comparison"""
        
        models = ['Word2Vec', 'Transformer']
        coverage_rates = [
            coverage_data['word2vec_coverage_rate'],
            coverage_data['transformer_coverage_rate']
        ]
        coverage_counts = [
            coverage_data['word2vec_coverage'],
            coverage_data['transformer_coverage']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=models,
            y=coverage_rates,
            text=[f"{rate:.1%}<br>({count}/{coverage_data['test_words']})" 
                  for rate, count in zip(coverage_rates, coverage_counts)],
            textposition='inside',
            marker_color=['red', 'blue']
        ))
        
        fig.update_layout(
            title={
                'text': 'Vocabulary Coverage Comparison',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title='Model',
            yaxis_title='Coverage Rate',
            yaxis=dict(tickformat='.0%'),
            width=600,
            height=500,
            template='plotly_white'
        )
        
        # Save chart
        filepath = os.path.join(self.results_dir, 'vocabulary_coverage.html')
        fig.write_html(filepath)
        print(f"Vocabulary coverage chart saved to {filepath}")
        
        return fig
    
    def create_dashboard(self, comparison_results: List[Dict], 
                        model_stats: Dict, coverage_data: Dict) -> go.Figure:
        """Create comprehensive dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Similarity Comparison', 'Improvement Analysis', 
                          'Performance Metrics', 'Vocabulary Coverage'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "bar"}]]
        )
        
        # Add similarity comparison
        word_pairs = [item['word_pair'] for item in comparison_results]
        word2vec_sims = [item['word2vec_similarity'] for item in comparison_results]
        transformer_sims = [item['transformer_similarity'] for item in comparison_results]
        
        fig.add_trace(
            go.Scatter(x=word_pairs, y=word2vec_sims, name='Word2Vec', 
                      line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=word_pairs, y=transformer_sims, name='Transformer',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add improvement analysis
        improvements = [item['improvement'] for item in comparison_results]
        colors = ['green' if x > 0 else 'red' for x in improvements]
        
        fig.add_trace(
            go.Bar(x=word_pairs, y=improvements, marker_color=colors,
                   name='Improvement', showlegend=False),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'SE Word Embeddings Analysis Dashboard',
                'x': 0.5,
                'font': {'size': 18}
            },
            width=1200,
            height=800,
            template='plotly_white'
        )
        
        # Save dashboard
        filepath = os.path.join(self.results_dir, 'dashboard.html')
        fig.write_html(filepath)
        print(f"Dashboard saved to {filepath}")
        
        return fig

