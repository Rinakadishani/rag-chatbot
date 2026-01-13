"""
Testing that the API key works
"""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('ANTHROPIC_API_KEY')

if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not found in .env file")
    print("Please check your .env file")
    exit(1)

print(f"✓ API key found: {api_key[:20]}...")

# Test the API
try:
    print("\nTesting API connection...")
    client = Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Say 'API key works!' if you can read this."}
        ]
    )
    
    print("\nSUCCESS! API Response:")
    print(message.content[0].text)
    print("\n🎉 Your API key is working perfectly!")
    
except Exception as e:
    print(f"\nERROR: API test failed")
    print(f"Error message: {str(e)}")
    print("\nPossible issues:")
    print("1. API key is incorrect")
    print("2. API key doesn't have permissions")
    print("3. Network connection issue")