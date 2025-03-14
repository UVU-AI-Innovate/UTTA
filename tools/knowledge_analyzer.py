#!/usr/bin/env python3
"""
Knowledge Base Analyzer

This script analyzes the knowledge base created from educational books and provides
insights into what the system has learned, how it categorizes knowledge, and how
that knowledge is used in generating teaching scenarios.

Features:
- Analyzes content distribution across knowledge categories
- Identifies key topics and concepts in the knowledge base
- Visualizes knowledge clusters and relationships
- Reports on most influential sources for scenario generation
- Tracks knowledge usage in generating scenarios
- Provides topic modeling of educational content

Usage:
    python knowledge_analyzer.py [--db-path DB_PATH] [--output-dir OUTPUT_DIR] [--format {text,html,json}]

Arguments:
    --db-path: Path to the knowledge database (default: ./knowledge.db or ./vector_index.index)
    --output-dir: Directory to save analysis results (default: ./analysis_results)
    --format: Output format for reports (default: text)
    --analyze-topics: Perform topic modeling on the knowledge base (default: False)
    --visualize: Generate visualizations (default: False)

Examples:
    python knowledge_analyzer.py --visualize
    python knowledge_analyzer.py --db-path custom_knowledge.db --analyze-topics
    python knowledge_analyzer.py --format html --output-dir ./my_analysis
"""

import os
import argparse
import json
import sqlite3
import numpy as np
import faiss
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

# Try to import optional dependencies
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class KnowledgeAnalyzer:
    """
    Analyzes the knowledge base and provides insights into its content and usage.
    """
    
    def __init__(self, db_path=None, output_dir="./analysis_results"):
        """
        Initialize the knowledge analyzer.
        
        Args:
            db_path: Path to the knowledge database
            output_dir: Directory to save analysis results
        """
        self.db_path = db_path
        self.output_dir = output_dir
        self.db_type = None  # Will be determined during connection
        self.chunks = []
        self.metadata = []
        self.categories = defaultdict(list)
        self.sources = defaultdict(list)
        self.topics = []
        self.effectiveness_scores = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to find the database if not specified
        if not db_path:
            for path in ["./knowledge.db", "./vector_index.index", "./vector_metadata.json"]:
                if os.path.exists(path):
                    self.db_path = path
                    break
        
        # Check that a database was found
        if not self.db_path:
            raise FileNotFoundError("No knowledge database found. Please specify the path with --db-path.")
        
        # Determine database type
        if self.db_path.endswith('.db'):
            self.db_type = "relational"
        elif self.db_path.endswith('.index'):
            self.db_type = "vector"
            # Check for metadata file
            metadata_path = self.db_path.replace('.index', '_metadata.json')
            if not os.path.exists(metadata_path):
                metadata_path = "vector_metadata.json"
                if not os.path.exists(metadata_path):
                    raise FileNotFoundError(f"Vector database metadata file not found: {metadata_path}")
            self.metadata_path = metadata_path
        else:
            raise ValueError(f"Unknown database type for file: {self.db_path}")
            
        logger.info(f"Database type detected: {self.db_type}")
        logger.info(f"Analysis results will be saved to: {self.output_dir}")
    
    def connect_and_load(self):
        """
        Connect to the knowledge database and load content for analysis.
        """
        if self.db_type == "relational":
            self._load_from_relational()
        elif self.db_type == "vector":
            self._load_from_vector()
        else:
            raise ValueError(f"Unknown database type: {self.db_type}")
        
        logger.info(f"Loaded {len(self.chunks)} chunks from {len(self.sources)} sources")
        
    def _load_from_relational(self):
        """Load content from a relational SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all chunks with their metadata
            cursor.execute('''
            SELECT c.id, c.text, c.metadata, c.category, c.effectiveness_score, 
                   d.source, d.title, d.author, d.subject
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            ''')
            
            for row in cursor.fetchall():
                chunk_id, text, metadata_json, category, score, source, title, author, subject = row
                
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError:
                    metadata = {}
                
                # Combine metadata
                metadata.update({
                    "source": source,
                    "title": title or "",
                    "author": author or "",
                    "subject": subject or ""
                })
                
                # Store chunk
                chunk = {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "category": category or "unknown",
                    "effectiveness_score": score or 0.0
                }
                
                self.chunks.append(chunk)
                self.categories[category or "unknown"].append(chunk)
                self.sources[source].append(chunk)
                self.effectiveness_scores[chunk_id] = score or 0.0
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading from relational database: {e}")
            raise
    
    def _load_from_vector(self):
        """Load content from a FAISS vector database with JSON metadata."""
        try:
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Process chunks
            for i, chunk_meta in enumerate(metadata.get("chunks", [])):
                # Get basic metadata
                chunk_id = chunk_meta.get("id", f"chunk_{i}")
                source = chunk_meta.get("source", "unknown")
                category = chunk_meta.get("category", "unknown")
                
                # Infer category if not present
                if category == "unknown" and "text" in chunk_meta:
                    category = self._infer_category(chunk_meta["text"])
                
                # Create chunk object
                chunk = {
                    "id": chunk_id,
                    "text": chunk_meta.get("text", ""),
                    "metadata": chunk_meta,
                    "category": category,
                    "effectiveness_score": chunk_meta.get("effectiveness_score", 0.0)
                }
                
                self.chunks.append(chunk)
                self.categories[category].append(chunk)
                self.sources[source].append(chunk)
                self.effectiveness_scores[chunk_id] = chunk_meta.get("effectiveness_score", 0.0)
                
        except Exception as e:
            logger.error(f"Error loading from vector database: {e}")
            raise
    
    def _infer_category(self, text):
        """Infer the category of a chunk based on its content."""
        text = text.lower()
        
        if any(term in text for term in ["classroom management", "behavior", "discipline", "classroom control"]):
            return "classroom_management"
        elif any(term in text for term in ["strategy", "teaching method", "instruction", "pedagogy"]):
            return "teaching_strategies"
        elif any(term in text for term in ["development", "learning style", "cognitive", "social emotional"]):
            return "student_development"
        else:
            return "general_education"
    
    def analyze(self):
        """
        Perform comprehensive analysis of the knowledge base.
        
        Returns:
            Dict with analysis results
        """
        logger.info("Starting knowledge base analysis...")
        
        # Connect and load data if not already done
        if not self.chunks:
            self.connect_and_load()
        
        # Perform various analyses
        category_stats = self._analyze_categories()
        source_stats = self._analyze_sources()
        content_stats = self._analyze_content()
        effectiveness_stats = self._analyze_effectiveness()
        
        # Compile results
        results = {
            "summary": {
                "total_chunks": len(self.chunks),
                "total_sources": len(self.sources),
                "total_categories": len(self.categories),
                "avg_chunk_length": sum(len(chunk["text"]) for chunk in self.chunks) / max(1, len(self.chunks)),
                "analyzed_at": datetime.now().isoformat()
            },
            "categories": category_stats,
            "sources": source_stats,
            "content": content_stats,
            "effectiveness": effectiveness_stats
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _analyze_categories(self):
        """Analyze the distribution and characteristics of knowledge categories."""
        category_stats = {}
        
        for category, chunks in self.categories.items():
            chunk_count = len(chunks)
            total_text = sum(len(chunk["text"]) for chunk in chunks)
            avg_chunk_length = total_text / chunk_count if chunk_count > 0 else 0
            
            # Get top sources for this category
            sources = Counter(chunk["metadata"].get("source", "unknown") for chunk in chunks)
            top_sources = sources.most_common(5)
            
            # Calculate effectiveness if available
            effectiveness_scores = [chunk.get("effectiveness_score", 0) for chunk in chunks if "effectiveness_score" in chunk]
            avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0
            
            # Store stats
            category_stats[category] = {
                "chunk_count": chunk_count,
                "total_text_length": total_text,
                "avg_chunk_length": avg_chunk_length,
                "top_sources": top_sources,
                "avg_effectiveness": avg_effectiveness,
                "percentage_of_total": (chunk_count / max(1, len(self.chunks))) * 100
            }
        
        return category_stats
    
    def _analyze_sources(self):
        """Analyze the sources of knowledge chunks."""
        source_stats = {}
        
        for source, chunks in self.sources.items():
            chunk_count = len(chunks)
            
            # Get categories for this source
            categories = Counter(chunk["category"] for chunk in chunks)
            top_categories = categories.most_common(5)
            
            # Get effectiveness if available
            effectiveness_scores = [chunk.get("effectiveness_score", 0) for chunk in chunks if "effectiveness_score" in chunk]
            avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0
            
            # Store stats
            source_stats[source] = {
                "chunk_count": chunk_count,
                "top_categories": top_categories,
                "avg_effectiveness": avg_effectiveness,
                "percentage_of_total": (chunk_count / max(1, len(self.chunks))) * 100
            }
        
        return source_stats
    
    def _analyze_content(self):
        """Analyze the content of knowledge chunks."""
        # Combine all text for overall analysis
        all_text = " ".join(chunk["text"] for chunk in self.chunks)
        
        # Extract keywords (simple approach)
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', all_text.lower())
        
        # Remove common stopwords
        stopwords = {"the", "and", "in", "of", "to", "a", "is", "that", "for", "with", 
                    "as", "on", "at", "this", "are", "be", "or", "by", "an", "they", 
                    "from", "their", "have", "was", "will", "not", "but", "what", "all", 
                    "were", "when", "can", "which", "there", "has", "more", "also"}
        filtered_words = [word for word in words if word not in stopwords]
        
        # Get most common words
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(50)
        
        # Analyze by category
        category_content = {}
        for category, chunks in self.categories.items():
            # Combine text for this category
            category_text = " ".join(chunk["text"] for chunk in chunks)
            
            # Extract keywords
            category_words = re.findall(r'\b[a-zA-Z]{3,15}\b', category_text.lower())
            filtered_category_words = [word for word in category_words if word not in stopwords]
            
            # Get most common words
            category_word_counts = Counter(filtered_category_words)
            category_top_words = category_word_counts.most_common(20)
            
            category_content[category] = {
                "top_words": category_top_words,
                "word_count": len(category_words),
                "unique_word_count": len(set(category_words))
            }
        
        return {
            "top_words": top_words,
            "word_count": len(words),
            "unique_word_count": len(set(words)),
            "by_category": category_content
        }
    
    def _analyze_effectiveness(self):
        """Analyze the effectiveness scores of knowledge chunks."""
        # Skip if no effectiveness scores
        if not self.effectiveness_scores:
            return {"available": False}
        
        # Calculate statistics
        scores = list(self.effectiveness_scores.values())
        
        # Get most effective chunks
        top_chunk_ids = sorted(self.effectiveness_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        top_chunks = []
        
        for chunk_id, score in top_chunk_ids:
            # Find the chunk
            chunk = next((c for c in self.chunks if str(c["id"]) == str(chunk_id)), None)
            if chunk:
                top_chunks.append({
                    "id": chunk_id,
                    "text": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
                    "source": chunk["metadata"].get("source", "unknown"),
                    "category": chunk["category"],
                    "effectiveness_score": score
                })
        
        # Analyze by category
        by_category = {}
        for category, chunks in self.categories.items():
            category_scores = [chunk.get("effectiveness_score", 0) for chunk in chunks if "effectiveness_score" in chunk]
            if not category_scores:
                continue
                
            by_category[category] = {
                "avg_score": sum(category_scores) / len(category_scores),
                "min_score": min(category_scores),
                "max_score": max(category_scores),
                "count": len(category_scores)
            }
        
        # Analyze by source
        by_source = {}
        for source, chunks in self.sources.items():
            source_scores = [chunk.get("effectiveness_score", 0) for chunk in chunks if "effectiveness_score" in chunk]
            if not source_scores:
                continue
                
            by_source[source] = {
                "avg_score": sum(source_scores) / len(source_scores),
                "min_score": min(source_scores),
                "max_score": max(source_scores),
                "count": len(source_scores)
            }
        
        return {
            "available": True,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "score_distribution": {
                "0.0-0.2": len([s for s in scores if 0.0 <= s < 0.2]),
                "0.2-0.4": len([s for s in scores if 0.2 <= s < 0.4]),
                "0.4-0.6": len([s for s in scores if 0.4 <= s < 0.6]),
                "0.6-0.8": len([s for s in scores if 0.6 <= s < 0.8]),
                "0.8-1.0": len([s for s in scores if 0.8 <= s <= 1.0])
            },
            "top_chunks": top_chunks,
            "by_category": by_category,
            "by_source": by_source
        }
    
    def _save_results(self, results):
        """Save analysis results to file."""
        output_file = os.path.join(self.output_dir, "knowledge_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Analysis results saved to: {output_file}")
    
    def generate_visualizations(self):
        """
        Generate visualizations of the knowledge base analysis.
        
        Requires matplotlib and optionally wordcloud and pandas.
        """
        if not self.chunks:
            self.connect_and_load()
            
        logger.info("Generating visualizations...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Category distribution pie chart
        self._visualize_category_distribution(plots_dir)
        
        # 2. Source distribution bar chart
        self._visualize_source_distribution(plots_dir)
        
        # 3. Word cloud (if available)
        if WORDCLOUD_AVAILABLE:
            self._visualize_word_cloud(plots_dir)
        else:
            logger.warning("WordCloud not available. Install wordcloud package for word cloud visualization.")
        
        # 4. Effectiveness score distribution (if available)
        if self.effectiveness_scores:
            self._visualize_effectiveness_scores(plots_dir)
            
        logger.info(f"Visualizations saved to: {plots_dir}")
    
    def _visualize_category_distribution(self, plots_dir):
        """Create a pie chart of knowledge distribution by category."""
        category_counts = {category: len(chunks) for category, chunks in self.categories.items()}
        
        plt.figure(figsize=(10, 6))
        plt.pie(
            category_counts.values(), 
            labels=category_counts.keys(), 
            autopct='%1.1f%%',
            startangle=140
        )
        plt.axis('equal')
        plt.title('Knowledge Distribution by Category')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "category_distribution.png"))
        plt.close()
    
    def _visualize_source_distribution(self, plots_dir):
        """Create a bar chart of knowledge distribution by source."""
        source_counts = {source: len(chunks) for source, chunks in self.sources.items()}
        
        # Sort by count
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top 10 sources for readability
        top_sources = sorted_sources[:10]
        
        plt.figure(figsize=(12, 6))
        plt.bar([s[0] for s in top_sources], [s[1] for s in top_sources])
        plt.xlabel('Source')
        plt.ylabel('Number of Chunks')
        plt.title('Top 10 Sources by Knowledge Chunk Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "source_distribution.png"))
        plt.close()
    
    def _visualize_word_cloud(self, plots_dir):
        """Create word clouds for overall content and by category."""
        # Overall word cloud
        all_text = " ".join(chunk["text"] for chunk in self.chunks)
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100
        ).generate(all_text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('Word Cloud of All Knowledge')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "word_cloud_all.png"))
        plt.close()
        
        # Word cloud by category
        for category, chunks in self.categories.items():
            if not chunks:
                continue
                
            category_text = " ".join(chunk["text"] for chunk in chunks)
            
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100
            ).generate(category_text)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f'Word Cloud for Category: {category}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"word_cloud_{category}.png"))
            plt.close()
    
    def _visualize_effectiveness_scores(self, plots_dir):
        """Create visualizations for effectiveness scores."""
        scores = list(self.effectiveness_scores.values())
        
        # Histogram of scores
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=10, alpha=0.7, color='blue')
        plt.xlabel('Effectiveness Score')
        plt.ylabel('Number of Chunks')
        plt.title('Distribution of Effectiveness Scores')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "effectiveness_histogram.png"))
        plt.close()
        
        # Average effectiveness by category
        if len(self.categories) > 1:
            category_avg_scores = {}
            for category, chunks in self.categories.items():
                category_scores = [chunk.get("effectiveness_score", 0) for chunk in chunks if "effectiveness_score" in chunk]
                if category_scores:
                    category_avg_scores[category] = sum(category_scores) / len(category_scores)
            
            plt.figure(figsize=(10, 6))
            plt.bar(category_avg_scores.keys(), category_avg_scores.values())
            plt.xlabel('Category')
            plt.ylabel('Average Effectiveness Score')
            plt.title('Average Effectiveness by Category')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "effectiveness_by_category.png"))
            plt.close()
    
    def perform_topic_modeling(self, num_topics=5, num_words=10):
        """
        Perform topic modeling on the knowledge base.
        
        Requires scikit-learn for NMF or LDA topic modeling.
        
        Args:
            num_topics: Number of topics to extract
            num_words: Number of words per topic to display
            
        Returns:
            Dict with topic modeling results
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available. Cannot perform topic modeling.")
            return {"error": "scikit-learn not available"}
        
        if not self.chunks:
            self.connect_and_load()
            
        logger.info(f"Performing topic modeling with {num_topics} topics...")
        
        # Prepare documents
        documents = [chunk["text"] for chunk in self.chunks]
        
        # Create TF-IDF representation
        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.95, 
            min_df=2,
            stop_words='english'
        )
        tfidf = tfidf_vectorizer.fit_transform(documents)
        
        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Perform NMF topic modeling
        nmf_model = NMF(n_components=num_topics, random_state=1)
        nmf_W = nmf_model.fit_transform(tfidf)
        nmf_H = nmf_model.components_
        
        # Get top words for each topic
        nmf_topics = []
        for topic_idx, topic in enumerate(nmf_H):
            top_indices = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_indices]
            nmf_topics.append({
                "id": topic_idx,
                "words": top_words,
                "prevalence": float(np.sum(nmf_W[:, topic_idx]) / np.sum(nmf_W))
            })
        
        # Try LDA topic modeling as well
        try:
            lda_model = LatentDirichletAllocation(
                n_components=num_topics, 
                max_iter=10,
                learning_method='online',
                random_state=0
            )
            lda_W = lda_model.fit_transform(tfidf)
            lda_H = lda_model.components_
            
            # Get top words for each topic
            lda_topics = []
            for topic_idx, topic in enumerate(lda_H):
                top_indices = topic.argsort()[:-num_words-1:-1]
                top_words = [feature_names[i] for i in top_indices]
                lda_topics.append({
                    "id": topic_idx,
                    "words": top_words,
                    "prevalence": float(np.sum(lda_W[:, topic_idx]) / np.sum(lda_W))
                })
        except:
            lda_topics = []
            logger.warning("LDA topic modeling failed. Only NMF results available.")
        
        # Save results
        topics_result = {
            "nmf_topics": nmf_topics,
            "lda_topics": lda_topics,
            "num_documents": len(documents),
            "vocab_size": len(feature_names)
        }
        
        # Save to file
        output_file = os.path.join(self.output_dir, "topic_modeling.json")
        with open(output_file, 'w') as f:
            json.dump(topics_result, f, indent=2)
        logger.info(f"Topic modeling results saved to: {output_file}")
        
        return topics_result
    
    def generate_summary_report(self, format="text"):
        """
        Generate a human-readable summary report of the knowledge base.
        
        Args:
            format: Output format (text, html, or markdown)
            
        Returns:
            The report as a string
        """
        if not self.chunks:
            self.connect_and_load()
            
        # Perform analysis if not already done
        analysis_file = os.path.join(self.output_dir, "knowledge_analysis.json")
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
        else:
            analysis = self.analyze()
        
        if format == "text":
            return self._generate_text_report(analysis)
        elif format == "html":
            return self._generate_html_report(analysis)
        elif format == "markdown":
            return self._generate_markdown_report(analysis)
        else:
            return self._generate_text_report(analysis)
    
    def _generate_text_report(self, analysis):
        """Generate a text report from analysis results."""
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("KNOWLEDGE BASE ANALYSIS REPORT".center(80))
        report.append("=" * 80)
        report.append("")
        
        # Summary
        summary = analysis["summary"]
        report.append("SUMMARY".center(80))
        report.append("-" * 80)
        report.append(f"Total Chunks: {summary['total_chunks']}")
        report.append(f"Total Sources: {summary['total_sources']}")
        report.append(f"Total Categories: {summary['total_categories']}")
        report.append(f"Average Chunk Length: {summary['avg_chunk_length']:.1f} characters")
        report.append(f"Analysis Date: {summary['analyzed_at']}")
        report.append("")
        
        # Categories
        report.append("CATEGORY DISTRIBUTION".center(80))
        report.append("-" * 80)
        for category, stats in analysis["categories"].items():
            report.append(f"Category: {category}")
            report.append(f"  Chunks: {stats['chunk_count']} ({stats['percentage_of_total']:.1f}% of total)")
            report.append(f"  Average Chunk Length: {stats['avg_chunk_length']:.1f} characters")
            report.append(f"  Top Sources:")
            for source, count in stats["top_sources"]:
                report.append(f"    - {source}: {count} chunks")
            report.append("")
        
        # Sources
        report.append("TOP SOURCES".center(80))
        report.append("-" * 80)
        
        # Sort sources by chunk count
        sorted_sources = sorted(
            analysis["sources"].items(), 
            key=lambda x: x[1]["chunk_count"], 
            reverse=True
        )
        
        # Show top 10 sources
        for source, stats in sorted_sources[:10]:
            report.append(f"Source: {source}")
            report.append(f"  Chunks: {stats['chunk_count']} ({stats['percentage_of_total']:.1f}% of total)")
            report.append(f"  Top Categories:")
            for category, count in stats["top_categories"]:
                report.append(f"    - {category}: {count} chunks")
            report.append("")
        
        # Content
        report.append("CONTENT ANALYSIS".center(80))
        report.append("-" * 80)
        report.append(f"Total Word Count: {analysis['content']['word_count']}")
        report.append(f"Unique Word Count: {analysis['content']['unique_word_count']}")
        report.append("Top Words:")
        for word, count in analysis["content"]["top_words"][:20]:
            report.append(f"  - {word}: {count}")
        report.append("")
        
        # Effectiveness
        if analysis["effectiveness"]["available"]:
            report.append("EFFECTIVENESS ANALYSIS".center(80))
            report.append("-" * 80)
            report.append(f"Average Effectiveness Score: {analysis['effectiveness']['avg_score']:.2f}")
            report.append(f"Min Score: {analysis['effectiveness']['min_score']:.2f}")
            report.append(f"Max Score: {analysis['effectiveness']['max_score']:.2f}")
            report.append("Score Distribution:")
            for range, count in analysis["effectiveness"]["score_distribution"].items():
                report.append(f"  - {range}: {count} chunks")
            report.append("")
            
            report.append("Top Effective Chunks:")
            for chunk in analysis["effectiveness"]["top_chunks"][:5]:
                report.append(f"  - [{chunk['source']}] ({chunk['effectiveness_score']:.2f}): {chunk['text']}")
            report.append("")
            
            report.append("Effectiveness by Category:")
            for category, stats in analysis["effectiveness"]["by_category"].items():
                report.append(f"  - {category}: {stats['avg_score']:.2f} average score ({stats['count']} chunks)")
            report.append("")
        
        # Final output
        output_file = os.path.join(self.output_dir, "knowledge_report.txt")
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        logger.info(f"Text report saved to: {output_file}")
        
        return "\n".join(report)
    
    def _generate_html_report(self, analysis):
        """Generate an HTML report from analysis results."""
        # Simple HTML template
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Knowledge Base Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .section { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
                .stats { display: flex; flex-wrap: wrap; }
                .stat-box { background: #f9f9f9; padding: 15px; margin: 10px; border-radius: 5px; flex: 1; min-width: 200px; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                table, th, td { border: 1px solid #ddd; }
                th, td { padding: 10px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="section">
                    <h1>Knowledge Base Analysis Report</h1>
                    <p>Analysis Date: {analysis_date}</p>
                </div>
                
                <div class="section">
                    <h2>Summary</h2>
                    <div class="stats">
                        <div class="stat-box">
                            <h3>Total Chunks</h3>
                            <p>{total_chunks}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Total Sources</h3>
                            <p>{total_sources}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Total Categories</h3>
                            <p>{total_categories}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Avg Chunk Length</h3>
                            <p>{avg_chunk_length:.1f} characters</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Category Distribution</h2>
                    <table>
                        <tr>
                            <th>Category</th>
                            <th>Chunks</th>
                            <th>Percentage</th>
                            <th>Avg Length</th>
                        </tr>
                        {category_rows}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Top Sources</h2>
                    <table>
                        <tr>
                            <th>Source</th>
                            <th>Chunks</th>
                            <th>Percentage</th>
                            <th>Top Categories</th>
                        </tr>
                        {source_rows}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Content Analysis</h2>
                    <div class="stats">
                        <div class="stat-box">
                            <h3>Total Words</h3>
                            <p>{word_count}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Unique Words</h3>
                            <p>{unique_word_count}</p>
                        </div>
                    </div>
                    
                    <h3>Top Words</h3>
                    <table>
                        <tr>
                            <th>Word</th>
                            <th>Count</th>
                        </tr>
                        {word_rows}
                    </table>
                </div>
                
                {effectiveness_section}
            </div>
        </body>
        </html>
        """
        
        # Format category rows
        category_rows = ""
        for category, stats in analysis["categories"].items():
            category_rows += f"""
            <tr>
                <td>{category}</td>
                <td>{stats['chunk_count']}</td>
                <td>{stats['percentage_of_total']:.1f}%</td>
                <td>{stats['avg_chunk_length']:.1f}</td>
            </tr>
            """
        
        # Format source rows (top 10)
        sorted_sources = sorted(
            analysis["sources"].items(), 
            key=lambda x: x[1]["chunk_count"], 
            reverse=True
        )[:10]
        
        source_rows = ""
        for source, stats in sorted_sources:
            top_categories = ", ".join([f"{cat} ({count})" for cat, count in stats["top_categories"][:3]])
            source_rows += f"""
            <tr>
                <td>{source}</td>
                <td>{stats['chunk_count']}</td>
                <td>{stats['percentage_of_total']:.1f}%</td>
                <td>{top_categories}</td>
            </tr>
            """
        
        # Format word rows
        word_rows = ""
        for word, count in analysis["content"]["top_words"][:20]:
            word_rows += f"""
            <tr>
                <td>{word}</td>
                <td>{count}</td>
            </tr>
            """
        
        # Format effectiveness section
        if analysis["effectiveness"]["available"]:
            effectiveness_section = f"""
            <div class="section">
                <h2>Effectiveness Analysis</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>Average Score</h3>
                        <p>{analysis['effectiveness']['avg_score']:.2f}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Min Score</h3>
                        <p>{analysis['effectiveness']['min_score']:.2f}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Max Score</h3>
                        <p>{analysis['effectiveness']['max_score']:.2f}</p>
                    </div>
                </div>
                
                <h3>Score Distribution</h3>
                <table>
                    <tr>
                        <th>Range</th>
                        <th>Count</th>
                    </tr>
            """
            
            for range, count in analysis["effectiveness"]["score_distribution"].items():
                effectiveness_section += f"""
                <tr>
                    <td>{range}</td>
                    <td>{count}</td>
                </tr>
                """
            
            effectiveness_section += """
                </table>
                
                <h3>Top Effective Chunks</h3>
                <table>
                    <tr>
                        <th>Source</th>
                        <th>Score</th>
                        <th>Text</th>
                    </tr>
            """
            
            for chunk in analysis["effectiveness"]["top_chunks"][:5]:
                effectiveness_section += f"""
                <tr>
                    <td>{chunk['source']}</td>
                    <td>{chunk['effectiveness_score']:.2f}</td>
                    <td>{chunk['text']}</td>
                </tr>
                """
            
            effectiveness_section += """
                </table>
                
                <h3>Effectiveness by Category</h3>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Avg Score</th>
                        <th>Chunks</th>
                    </tr>
            """
            
            for category, stats in analysis["effectiveness"]["by_category"].items():
                effectiveness_section += f"""
                <tr>
                    <td>{category}</td>
                    <td>{stats['avg_score']:.2f}</td>
                    <td>{stats['count']}</td>
                </tr>
                """
            
            effectiveness_section += """
                </table>
            </div>
            """
        else:
            effectiveness_section = ""
        
        # Format template
        formatted_html = html.format(
            analysis_date=analysis["summary"]["analyzed_at"],
            total_chunks=analysis["summary"]["total_chunks"],
            total_sources=analysis["summary"]["total_sources"],
            total_categories=analysis["summary"]["total_categories"],
            avg_chunk_length=analysis["summary"]["avg_chunk_length"],
            category_rows=category_rows,
            source_rows=source_rows,
            word_count=analysis["content"]["word_count"],
            unique_word_count=analysis["content"]["unique_word_count"],
            word_rows=word_rows,
            effectiveness_section=effectiveness_section
        )
        
        # Save HTML report
        output_file = os.path.join(self.output_dir, "knowledge_report.html")
        with open(output_file, 'w') as f:
            f.write(formatted_html)
        logger.info(f"HTML report saved to: {output_file}")
        
        return formatted_html
    
    def _generate_markdown_report(self, analysis):
        """Generate a Markdown report from analysis results."""
        report = []
        
        # Header
        report.append("# Knowledge Base Analysis Report")
        report.append(f"Analysis Date: {analysis['summary']['analyzed_at']}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Total Chunks**: {analysis['summary']['total_chunks']}")
        report.append(f"- **Total Sources**: {analysis['summary']['total_sources']}")
        report.append(f"- **Total Categories**: {analysis['summary']['total_categories']}")
        report.append(f"- **Average Chunk Length**: {analysis['summary']['avg_chunk_length']:.1f} characters")
        report.append("")
        
        # Categories
        report.append("## Category Distribution")
        report.append("| Category | Chunks | Percentage | Avg Length |")
        report.append("| --- | --- | --- | --- |")
        for category, stats in analysis["categories"].items():
            report.append(f"| {category} | {stats['chunk_count']} | {stats['percentage_of_total']:.1f}% | {stats['avg_chunk_length']:.1f} |")
        report.append("")
        
        # Sources
        report.append("## Top Sources")
        report.append("| Source | Chunks | Percentage | Top Categories |")
        report.append("| --- | --- | --- | --- |")
        
        # Sort sources by chunk count
        sorted_sources = sorted(
            analysis["sources"].items(), 
            key=lambda x: x[1]["chunk_count"], 
            reverse=True
        )
        
        # Show top 10 sources
        for source, stats in sorted_sources[:10]:
            top_categories = ", ".join([f"{cat} ({count})" for cat, count in stats["top_categories"][:2]])
            report.append(f"| {source} | {stats['chunk_count']} | {stats['percentage_of_total']:.1f}% | {top_categories} |")
        report.append("")
        
        # Content
        report.append("## Content Analysis")
        report.append(f"- **Total Word Count**: {analysis['content']['word_count']}")
        report.append(f"- **Unique Word Count**: {analysis['content']['unique_word_count']}")
        report.append("")
        
        report.append("### Top Words")
        report.append("| Word | Count |")
        report.append("| --- | --- |")
        for word, count in analysis["content"]["top_words"][:20]:
            report.append(f"| {word} | {count} |")
        report.append("")
        
        # Effectiveness
        if analysis["effectiveness"]["available"]:
            report.append("## Effectiveness Analysis")
            report.append(f"- **Average Score**: {analysis['effectiveness']['avg_score']:.2f}")
            report.append(f"- **Min Score**: {analysis['effectiveness']['min_score']:.2f}")
            report.append(f"- **Max Score**: {analysis['effectiveness']['max_score']:.2f}")
            report.append("")
            
            report.append("### Score Distribution")
            report.append("| Range | Count |")
            report.append("| --- | --- |")
            for range, count in analysis["effectiveness"]["score_distribution"].items():
                report.append(f"| {range} | {count} |")
            report.append("")
            
            report.append("### Top Effective Chunks")
            for chunk in analysis["effectiveness"]["top_chunks"][:5]:
                report.append(f"- **[{chunk['source']}]** ({chunk['effectiveness_score']:.2f}): {chunk['text']}")
            report.append("")
            
            report.append("### Effectiveness by Category")
            report.append("| Category | Avg Score | Chunks |")
            report.append("| --- | --- | --- |")
            for category, stats in analysis["effectiveness"]["by_category"].items():
                report.append(f"| {category} | {stats['avg_score']:.2f} | {stats['count']} |")
            report.append("")
        
        # Save MD report
        output_file = os.path.join(self.output_dir, "knowledge_report.md")
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        logger.info(f"Markdown report saved to: {output_file}")
        
        return "\n".join(report)
    
    def analyze_scenario_generation(self, num_scenarios=10):
        """
        Analyze how knowledge is used in scenario generation.
        
        Generates a specified number of scenarios and analyzes which
        knowledge chunks are used most frequently, which are most effective,
        and how knowledge is distributed across scenarios.
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            Dict with scenario generation analysis
        """
        try:
            from scenario_generator import ClassroomScenarioGenerator
            from llm_handler import LLMInterface
            from vector_database import VectorDatabase
        except ImportError:
            logger.error("Required modules not found: scenario_generator, llm_handler, or vector_database")
            return {"error": "Required modules not found"}
            
        logger.info(f"Analyzing scenario generation with {num_scenarios} scenarios...")
        
        # Initialize components
        try:
            vector_db = VectorDatabase()
            llm = LLMInterface()
            scenario_generator = ClassroomScenarioGenerator(vector_db, llm)
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return {"error": f"Error initializing components: {e}"}
        
        # Generate scenarios
        scenarios = []
        knowledge_used = []
        grade_levels = ["elementary", "middle", "high"]
        subjects = ["math", "reading", "science", "social studies", None]
        challenge_types = ["attention issues", "disruptive behavior", "motivation", "conflict", None]
        
        try:
            for i in range(num_scenarios):
                grade = random.choice(grade_levels)
                subject = random.choice(subjects)
                challenge = random.choice(challenge_types)
                
                logger.info(f"Generating scenario {i+1}/{num_scenarios}: {grade} {subject} {challenge}")
                
                scenario = scenario_generator.generate_scenario(
                    grade_level=grade,
                    subject=subject,
                    challenge_type=challenge
                )
                
                scenarios.append({
                    "id": i,
                    "parameters": {
                        "grade_level": grade,
                        "subject": subject,
                        "challenge_type": challenge
                    },
                    "knowledge_sources": scenario.get("knowledge_sources", []),
                    "scenario_length": len(scenario.get("scenario", ""))
                })
                
                # Track knowledge used
                for source in scenario.get("knowledge_sources", []):
                    knowledge_used.append(source["id"])
        except Exception as e:
            logger.error(f"Error generating scenarios: {e}")
            # Continue with analysis of any scenarios generated so far
        
        # Analyze knowledge usage
        knowledge_counts = Counter(knowledge_used)
        
        # Get details for most used knowledge
        most_used_knowledge = []
        for chunk_id, count in knowledge_counts.most_common(10):
            try:
                conn = sqlite3.connect(vector_db.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT text, metadata FROM chunks WHERE id = ?", 
                    (chunk_id,)
                )
                result = cursor.fetchone()
                if result:
                    text, metadata_json = result
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    most_used_knowledge.append({
                        "id": chunk_id,
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "source": metadata.get("source", "unknown"),
                        "usage_count": count,
                        "usage_percentage": (count / len(scenarios)) * 100
                    })
                conn.close()
            except Exception as e:
                logger.error(f"Error retrieving chunk {chunk_id}: {e}")
        
        # Analyze knowledge distribution
        knowledge_per_scenario = [len(s["knowledge_sources"]) for s in scenarios]
        avg_knowledge_per_scenario = sum(knowledge_per_scenario) / len(knowledge_per_scenario) if knowledge_per_scenario else 0
        
        # Calculate unique knowledge usage
        unique_knowledge = len(set(knowledge_used))
        knowledge_reuse_ratio = len(knowledge_used) / unique_knowledge if unique_knowledge > 0 else 0
        
        # Generate results
        results = {
            "scenarios_generated": len(scenarios),
            "unique_knowledge_chunks_used": unique_knowledge,
            "total_knowledge_usage": len(knowledge_used),
            "knowledge_reuse_ratio": knowledge_reuse_ratio,
            "avg_knowledge_per_scenario": avg_knowledge_per_scenario,
            "most_used_knowledge": most_used_knowledge,
            "knowledge_distribution": {
                "min_per_scenario": min(knowledge_per_scenario) if knowledge_per_scenario else 0,
                "max_per_scenario": max(knowledge_per_scenario) if knowledge_per_scenario else 0,
                "distribution": Counter(knowledge_per_scenario)
            },
            "parameter_impact": {
                "grade_level": {},
                "subject": {},
                "challenge_type": {}
            }
        }
        
        # Analyze impact of parameters on knowledge usage
        for param in ["grade_level", "subject", "challenge_type"]:
            param_knowledge = defaultdict(list)
            
            for scenario in scenarios:
                param_value = scenario["parameters"].get(param)
                if param_value:
                    for source in scenario["knowledge_sources"]:
                        param_knowledge[param_value].append(source["id"])
            
            for value, knowledge_ids in param_knowledge.items():
                results["parameter_impact"][param][value] = {
                    "count": len(knowledge_ids),
                    "unique": len(set(knowledge_ids)),
                    "avg_per_scenario": len(knowledge_ids) / len([s for s in scenarios if s["parameters"].get(param) == value])
                }
        
        # Save results
        output_file = os.path.join(self.output_dir, "scenario_generation_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Scenario generation analysis saved to: {output_file}")
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the knowledge base from educational books.")
    parser.add_argument("--db-path", help="Path to the knowledge database")
    parser.add_argument("--output-dir", default="./analysis_results", help="Directory to save analysis results")
    parser.add_argument("--format", choices=["text", "html", "markdown"], default="text", help="Output format for reports")
    parser.add_argument("--analyze-topics", action="store_true", help="Perform topic modeling on the knowledge base")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--analyze-scenarios", action="store_true", help="Analyze scenario generation")
    parser.add_argument("--num-scenarios", type=int, default=10, help="Number of scenarios to generate for analysis")
    parser.add_argument("--num-topics", type=int, default=5, help="Number of topics for topic modeling")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = KnowledgeAnalyzer(db_path=args.db_path, output_dir=args.output_dir)
        
        # Load data
        analyzer.connect_and_load()
        
        # Perform analysis
        analysis_results = analyzer.analyze()
        
        # Generate report
        report = analyzer.generate_summary_report(format=args.format)
        print(f"Analysis report generated in format: {args.format}")
        
        # Generate visualizations if requested
        if args.visualize:
            analyzer.generate_visualizations()
            print("Visualizations generated")
        
        # Perform topic modeling if requested
        if args.analyze_topics:
            if SKLEARN_AVAILABLE:
                topic_results = analyzer.perform_topic_modeling(num_topics=args.num_topics)
                print(f"Topic modeling completed with {args.num_topics} topics")
            else:
                print("Topic modeling requires scikit-learn. Please install it with: pip install scikit-learn")
        
        # Analyze scenario generation if requested
        if args.analyze_scenarios:
            scenario_results = analyzer.analyze_scenario_generation(num_scenarios=args.num_scenarios)
            print(f"Scenario generation analysis completed with {args.num_scenarios} scenarios")
        
        print(f"Analysis complete. Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 