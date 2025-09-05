#!/usr/bin/env python3
"""
Setup script to download required NLTK data for deployment
"""
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download required NLTK data"""
    datasets = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'vader_lexicon'
    ]
    
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
            print(f"✅ Downloaded {dataset}")
        except Exception as e:
            print(f"⚠️ Failed to download {dataset}: {e}")

if __name__ == "__main__":
    download_nltk_data()
    print("🎉 NLTK setup complete!")
