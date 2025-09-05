#!/usr/bin/env python3
"""
Test script to verify OpenAI integration works properly
"""
import os
import sys
sys.path.append('src')

from models.genai_analyzer import GenAIDocumentAnalyzer

def test_openai_integration():
    print("Testing OpenAI Integration...")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✓ API Key found: {'*' * 20}{api_key[-4:]}")
    else:
        print("❌ No API key found")
        return
    
    # Initialize analyzer
    try:
        analyzer = GenAIDocumentAnalyzer()
        print(f"✓ GenAI Analyzer initialized")
        print(f"  - OpenAI Available: {analyzer.openai_available}")
        print(f"  - OpenAI Client: {analyzer.openai_client is not None}")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return
    
    # Test API status
    try:
        status = analyzer.check_api_status()
        print(f"✓ API Status check completed:")
        for key, value in status.items():
            print(f"  - {key}: {value}")
    except Exception as e:
        print(f"❌ Error checking API status: {e}")
        return
    
    # Test Q&A
    if analyzer.openai_available:
        test_text = """
        This is a test document about artificial intelligence and machine learning.
        It discusses the applications of AI in various fields including healthcare,
        finance, and autonomous vehicles. The document emphasizes the importance
        of responsible AI development and ethical considerations.
        """
        
        test_question = "What are the main applications of AI mentioned?"
        
        try:
            print(f"✓ Testing Q&A with question: '{test_question}'")
            result = analyzer.answer_question(test_text, test_question, "openai")
            print(f"✓ Answer received: {result['answer'][:100]}...")
            print(f"  - Method: {result.get('method', 'unknown')}")
            print(f"  - Confidence: {result.get('confidence', 0)}")
        except Exception as e:
            print(f"❌ Error in Q&A test: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_openai_integration()
