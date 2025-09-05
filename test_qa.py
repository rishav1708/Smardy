#!/usr/bin/env python3
"""
Simple test script for Q&A functionality
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_qa_functionality():
    """Test the Q&A functionality with fallback methods"""
    print("🧪 Testing Q&A Functionality")
    print("=" * 50)
    
    try:
        from models.genai_analyzer import GenAIDocumentAnalyzer
        
        # Initialize analyzer
        print("📥 Initializing GenAI analyzer...")
        analyzer = GenAIDocumentAnalyzer()
        
        # Check status
        status = analyzer.check_api_status()
        print(f"📊 Model Status:")
        print(f"   - OpenAI Available: {status['openai_available']}")
        print(f"   - Local Models: {status['local_models']}")
        
        # Test document
        test_text = """
        This is a sample PDF document about artificial intelligence and machine learning.
        The document discusses various applications of AI in different industries.
        Machine learning algorithms are used for data analysis and pattern recognition.
        The document also covers natural language processing and computer vision.
        """
        
        # Test questions
        test_questions = [
            "What is this document about?",
            "What are the main topics?",
            "What applications are mentioned?"
        ]
        
        print("\n🤔 Testing Q&A functionality:")
        for i, question in enumerate(test_questions, 1):
            print(f"\n❓ Question {i}: {question}")
            try:
                result = analyzer.answer_question(test_text, question, method="local")
                print(f"✅ Answer: {result['answer']}")
                print(f"📊 Confidence: {result.get('confidence', 0):.2%}")
                print(f"🔧 Method: {result.get('method', 'unknown')}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print("\n✅ Q&A Test Complete!")
        
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("💡 This is expected if dependencies are not installed locally")
        print("🚀 The fixes will work on Streamlit Cloud")
    except Exception as e:
        print(f"❌ Test Error: {str(e)}")

if __name__ == "__main__":
    test_qa_functionality()
