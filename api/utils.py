"""
Utility Functions for Text Processing

This module provides helper functions for text manipulation and formatting.
Used primarily for processing user input and formatting workflow results.

Functions:
    - split_into_sentences: Basic sentence segmentation using regex
    - clean_text: Text normalization and whitespace cleanup
    - format_qa_pairs: Format question-answer pairs for display
    
Features:
    - Regex-based text processing
    - Consistent text formatting
    - Support for Q&A pair formatting
"""

import re
from typing import List

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using basic regex.
    
    Uses punctuation marks (., !, ?) to identify sentence boundaries.
    Removes empty sentences and strips whitespace.
    
    Args:
        text: Input text to split into sentences
        
    Returns:
        List of cleaned sentences
        
    Note:
        Basic implementation - could be enhanced with NLP libraries for better accuracy
    """
    # Basic sentence splitting - can be enhanced with more sophisticated NLP
    sentences = re.split(r'[.!?]+', text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Normalizes whitespace by replacing multiple consecutive whitespace
    characters with single spaces and removing leading/trailing whitespace.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned and normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def format_qa_pairs(pairs: List[tuple]) -> str:
    """
    Format question-answer pairs into a standardized display format.
    
    Creates formatted output with "Question:" and "Answer:" labels,
    separated by empty lines for readability.
    
    Args:
        pairs: List of (question, answer) tuples
        
    Returns:
        Formatted string with Q&A pairs
        
    Example:
        >>> pairs = [("What is AI?", "Artificial Intelligence")]
        >>> format_qa_pairs(pairs)
        'Question: What is AI?\nAnswer: Artificial Intelligence'
    """
    formatted_lines = []
    
    for question, answer in pairs:
        formatted_lines.append(f"Question: {question}")
        formatted_lines.append(f"Answer: {answer}")
        formatted_lines.append("")  # Empty line separator
    
    return "\n".join(formatted_lines).strip()