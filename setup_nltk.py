#!/usr/bin/env python3
"""
Setup script to download required NLTK data for deployment
"""
import nltk
import ssl
import os

# Fix SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure NLTK data directory exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download required NLTK data
nltk_downloads = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'vader_lexicon'
]

print("Setting up NLTK data...")
for item in nltk_downloads:
    try:
        nltk.download(item, quiet=True)
        print(f"✅ Downloaded {item}")
    except Exception as e:
        print(f"⚠️ Error downloading {item}: {e}")

print("🎉 NLTK setup complete!")

if __name__ == "__main__":
    download_nltk_data()
    print("🎉 NLTK setup complete!")
