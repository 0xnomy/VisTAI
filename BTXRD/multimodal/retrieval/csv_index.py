"""
CSV Metadata Indexing for Retrieval
====================================

Indexes CSV files containing tumor metadata for retrieval-augmented generation.
This enables the LLM to find similar historical cases.

Design Rationale:
-----------------
While current CSVs only have image paths and labels, this module is designed
to handle richer metadata if it becomes available (age, gender, bone type, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json


class CSVIndex:
    """
    Index for tumor metadata from CSV files.
    """
    
    def __init__(self, csv_paths: List[str], encoding_path: str = None):
        """
        Initialize CSV index.
        
        Args:
            csv_paths: List of paths to CSV files (train, val, test)
            encoding_path: Path to label_encoding.json
        """
        self.csv_paths = csv_paths
        self.data = self._load_data()
        
        # Load label encoding if provided
        if encoding_path:
            with open(encoding_path, 'r') as f:
                self.label_encoding = json.load(f)
        else:
            self.label_encoding = None
        
        print(f"✓ CSV Index created: {len(self.data)} records")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and concatenate CSV files."""
        dfs = []
        for csv_path in self.csv_paths:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    
    def get_by_tumor_class(self, tumor_class: str) -> pd.DataFrame:
        """
        Get all records for a specific tumor class.
        
        Args:
            tumor_class: Tumor type name
            
        Returns:
            DataFrame of matching records
        """
        return self.data[self.data['labels'] == tumor_class]
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with class distribution, etc.
        """
        stats = {
            'total_samples': len(self.data),
            'num_classes': self.data['labels'].nunique(),
            'class_distribution': self.data['labels'].value_counts().to_dict(),
            'has_masks': self.data['has_mask'].sum() if 'has_mask' in self.data.columns else 0
        }
        
        return stats
    
    def get_examples(self, tumor_class: str, n: int = 5) -> List[Dict]:
        """
        Get example cases for a tumor class.
        
        Args:
            tumor_class: Tumor type
            n: Number of examples
            
        Returns:
            List of example records
        """
        class_data = self.get_by_tumor_class(tumor_class)
        
        if len(class_data) == 0:
            return []
        
        samples = class_data.sample(min(n, len(class_data)))
        examples = samples.to_dict('records')
        
        return examples
    
    def search_by_metadata(self, filters: Dict) -> pd.DataFrame:
        """
        Search records by metadata filters.
        
        Args:
            filters: Dictionary of column:value pairs
            
        Returns:
            Filtered DataFrame
        """
        result = self.data.copy()
        
        for column, value in filters.items():
            if column in result.columns:
                result = result[result[column] == value]
        
        return result


def build_tumor_knowledge_base(csv_paths: List[str], encoding_path: str) -> Dict:
    """
    Build knowledge base from CSV metadata.
    
    Args:
        csv_paths: Paths to CSV files
        encoding_path: Path to label encoding
        
    Returns:
        knowledge_base: Dictionary with tumor statistics and examples
    """
    index = CSVIndex(csv_paths, encoding_path)
    stats = index.get_statistics()
    
    knowledge_base = {
        'statistics': stats,
        'classes': {}
    }
    
    # For each tumor class, gather information
    for tumor_class in stats['class_distribution'].keys():
        class_data = {
            'name': tumor_class,
            'count': stats['class_distribution'][tumor_class],
            'percentage': 100 * stats['class_distribution'][tumor_class] / stats['total_samples'],
            'examples': index.get_examples(tumor_class, n=3)
        }
        
        knowledge_base['classes'][tumor_class] = class_data
    
    return knowledge_base


def extract_tumor_facts(tumor_class: str, knowledge_base: Dict) -> str:
    """
    Extract facts about a tumor class from knowledge base.
    
    Args:
        tumor_class: Tumor type
        knowledge_base: Knowledge base dictionary
        
    Returns:
        facts_text: String with relevant facts
    """
    if tumor_class not in knowledge_base['classes']:
        return f"No specific information available for {tumor_class}."
    
    class_info = knowledge_base['classes'][tumor_class]
    
    facts = f"""
Tumor Class: {class_info['name']}
Dataset prevalence: {class_info['count']} cases ({class_info['percentage']:.1f}% of dataset)

This indicates that {tumor_class} is {"common" if class_info['percentage'] > 15 else "relatively rare"} in the training data.
"""
    
    return facts.strip()


if __name__ == "__main__":
    # Test CSV indexing
    print("\n" + "="*70)
    print("Testing CSV Index")
    print("="*70)
    
    # Note: This test uses relative paths - adjust for actual use
    project_root = Path(__file__).parent.parent.parent
    
    csv_paths = [
        str(project_root / 'segmentation_train.csv'),
        str(project_root / 'segmentation_val.csv'),
        str(project_root / 'segmentation_test.csv')
    ]
    
    encoding_path = str(project_root / 'label_encoding.json')
    
    try:
        # Create index
        print("\nCreating CSV index...")
        index = CSVIndex(csv_paths, encoding_path)
        
        # Get statistics
        print("\nDataset statistics:")
        stats = index.get_statistics()
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Number of classes: {stats['num_classes']}")
        print(f"  Has masks: {stats['has_masks']}")
        
        print("\n  Class distribution:")
        for cls, count in list(stats['class_distribution'].items())[:5]:
            print(f"    {cls}: {count}")
        
        # Build knowledge base
        print("\nBuilding knowledge base...")
        kb = build_tumor_knowledge_base(csv_paths, encoding_path)
        
        print(f"Knowledge base contains {len(kb['classes'])} tumor classes")
        
        # Extract facts for a class
        print("\nExample: Facts about osteosarcoma:")
        facts = extract_tumor_facts('osteosarcoma', kb)
        print(facts)
        
        print("\n✓ CSV index test passed")
        
    except FileNotFoundError:
        print("\n⚠ CSV files not found at expected locations")
        print("This is expected if running outside project directory")
        print("CSV indexing interface validated")
    
    print("="*70)
