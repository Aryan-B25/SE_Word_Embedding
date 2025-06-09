"""
Data Collection Module for SE Word Embeddings
"""

import os
import json
import requests
import wikipedia
from typing import List, Dict
import time

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
            "Software design patterns",
            "Agile software development",
            "Object-oriented programming",
            "Software debugging"
        ]
        
    def collect_se_pages(self, max_pages_per_category: int = 20) -> List[str]:
        """Collect software engineering pages using Wikipedia API"""
        all_texts = []
        collected_titles = set()
        
        print(f"Collecting SE pages from {len(self.se_categories)} categories...")
        
        for i, category in enumerate(self.se_categories):
            print(f"Processing category {i+1}/{len(self.se_categories)}: {category}")
            
            try:
                # Search for pages in this category
                search_results = wikipedia.search(category, results=max_pages_per_category)
                
                for title in search_results[:max_pages_per_category]:
                    if title in collected_titles:
                        continue
                        
                    try:
                        page = wikipedia.page(title)
                        
                        # Filter for SE-related content
                        if self._is_se_related(page.content):
                            all_texts.append({
                                'title': title,
                                'content': page.content,
                                'url': page.url,
                                'category': category
                            })
                            collected_titles.add(title)
                            print(f"  ✓ Collected: {title}")
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                    except (wikipedia.exceptions.DisambiguationError, 
                           wikipedia.exceptions.PageError) as e:
                        print(f"  ✗ Error with {title}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error collecting from category {category}: {e}")
                continue
                
        print(f"Total collected: {len(all_texts)} unique SE-related documents")
        return all_texts
    
    def _is_se_related(self, text: str) -> bool:
        """Check if text is software engineering related"""
        se_keywords = [
            'software', 'programming', 'development', 'algorithm', 'code',
            'testing', 'debugging', 'architecture', 'design pattern',
            'requirements', 'engineering', 'computer science', 'application',
            'system', 'framework', 'library', 'api', 'database'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in se_keywords if keyword in text_lower)
        
        # Require at least 3 SE keywords and minimum length
        return keyword_count >= 3 and len(text) > 500
    
    def save_collected_data(self, data: List[Dict], filename: str = "collected_data.json"):
        """Save collected data to JSON file"""
        filepath = os.path.join("results", filename)
        os.makedirs("results", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {filepath}")
    
    def load_collected_data(self, filename: str = "collected_data.json") -> List[Dict]:
        """Load previously collected data"""
        filepath = os.path.join("results", filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} documents from {filepath}")
            return data
        else:
            print(f"No existing data found at {filepath}")
            return []

class SampleDataGenerator:
    """Generate sample SE data for quick testing"""
    
    @staticmethod
    def get_sample_se_texts() -> List[str]:
        """Get sample SE texts for demonstration"""
        return [
            "Software engineering is the systematic application of engineering approaches to the development of software. It involves the use of principles from computer science, engineering, and mathematical analysis to design, develop, test, and maintain software systems.",
            
            "Object-oriented programming is a programming paradigm based on the concept of objects, which can contain data and code. Data is in the form of fields, and code is in the form of procedures. Classes define the structure and behavior of objects.",
            
            "Unit testing is a software testing method by which individual units of source code are tested to determine whether they are fit for use. A unit is the smallest testable part of any software application.",
            
            "Agile software development comprises various approaches to software development under which requirements and solutions evolve through the collaborative effort of self-organizing and cross-functional teams.",
            
            "Design patterns are reusable solutions to commonly occurring problems in software design. They represent best practices and provide a common vocabulary for developers to communicate design decisions.",
            
            "Version control systems are tools that help manage changes to source code over time. They keep track of every modification to the code in a special kind of database called a repository.",
            
            "Code review is a systematic examination of computer source code intended to find bugs and improve the overall quality of software. Reviews are done in various forms such as pair programming and formal inspections.",
            
            "Software architecture refers to the fundamental structures of a software system and the discipline of creating such structures. It involves making structural choices that are costly to change once implemented.",
            
            "Requirements engineering is the process of defining, documenting, and maintaining requirements in the engineering design process. It involves elicitation, analysis, specification, and validation of requirements.",
            
            "Debugging is the process of finding and resolving defects or problems within a computer program that prevent correct operation. It involves identifying the source of errors and fixing them.",
            
            "Continuous integration is a development practice where developers integrate code into a shared repository frequently. Each integration is verified by an automated build and automated tests.",
            
            "Software testing is an investigation conducted to provide stakeholders with information about the quality of the software product or service under test. It involves execution of software components.",
            
            "Refactoring is a disciplined technique for restructuring an existing body of code, altering its internal structure without changing its external behavior. It improves the design of existing code.",
            
            "API design involves creating application programming interfaces that are easy to use, understand, and maintain. Good API design follows principles of consistency, simplicity, and clear documentation.",
            
            "Database design is the organization of data according to a database model. It involves determining what data must be stored and how the data elements interrelate to create an efficient database structure."
        ]
    
    @staticmethod
    def get_se_vocabulary() -> List[str]:
        """Get SE-specific vocabulary for testing"""
        return [
            'software', 'engineering', 'programming', 'development', 'testing',
            'debugging', 'algorithm', 'architecture', 'design', 'pattern',
            'class', 'object', 'method', 'function', 'variable',
            'bug', 'error', 'exception', 'interface', 'implementation',
            'framework', 'library', 'api', 'database', 'system',
            'application', 'module', 'component', 'service', 'client',
            'server', 'protocol', 'network', 'security', 'performance',
            'scalability', 'maintainability', 'reliability', 'usability',
            'requirements', 'specification', 'documentation', 'version',
            'repository', 'commit', 'branch', 'merge', 'deployment'
        ]

