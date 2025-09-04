#!/bin/bash

# Smart Document Analyzer - Quick Start Script
# Author: Rishav Kant, Birla Institute of Technology, Mesra

set -e

echo "🚀 Starting Smart Document Analyzer Setup..."
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
echo "📥 Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
print('✅ NLTK data downloaded successfully')
"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "📝 Please edit .env file to add your API keys (optional)"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📖 Next steps:"
echo "1. (Optional) Edit .env file to add OpenAI API key for GenAI features"
echo "2. Run the application with: streamlit run app.py"
echo ""
echo "🌐 The app will open at: http://localhost:8501"
echo ""
echo "🐳 Alternative: Use Docker with 'docker-compose up --build'"
echo ""
echo "📄 For more information, see README.md"
echo ""

# Ask if user wants to start the app
read -p "🚀 Would you like to start the application now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎯 Starting Smart Document Analyzer..."
    streamlit run app.py
fi
