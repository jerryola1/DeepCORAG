import os
import sys
import json
import httpx
from typing import List, Dict, Any, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import openai
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage

print(f"API Key starts with: {os.getenv('DEEPSEEK_API_KEY')[:5]}...")

class DeepSeekLLM:
    def __init__(self, temperature: float = 0.7):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.temperature = temperature
        self.messages = []
        self.total_tokens_used = 0

    def _truncate_text(self, text: str, max_length: int = 4000) -> str:
        """Truncate text to prevent token limit issues"""
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def _make_api_call(self, messages: List[Dict[str, str]]) -> Any:
        """Make API call following DeepSeek documentation"""
        try:
            # Add system message at the start
            full_messages = [
                {
                    "role": "system",
                    "content": "You are a document analysis expert. When given context from a document and a question, provide a clear and focused answer based ONLY on the provided context. If the context doesn't contain enough information to answer the question, say so. Keep your answers concise and directly related to the question asked."
                }
            ]
            
            # Process and truncate user messages
            for msg in messages:
                if msg["role"] == "user":
                    msg["content"] = self._truncate_text(msg["content"])
                full_messages.append(msg)
            
            # Print messages for debugging
            print("\nSending request to DeepSeek API...")
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=2048,  # Reduced for safety
                stream=False
            )
            
            if not hasattr(response, 'choices') or not response.choices:
                print(f"Invalid response structure: {response}")
                raise RuntimeError("Invalid response from DeepSeek API")
                
            return response
            
        except httpx.HTTPError as e:
            print(f"HTTP Error: {str(e)}")
            raise RuntimeError(f"HTTP Error: {str(e)}")
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                raise RuntimeError("Rate limit exceeded. Please wait before trying again.")
            elif "authentication" in error_msg:
                raise RuntimeError("Authentication failed. Please check your API key.")
            elif "insufficient balance" in error_msg:
                raise RuntimeError("Insufficient balance in your DeepSeek account.")
            else:
                print(f"API Error: {str(e)}")
                raise RuntimeError(f"DeepSeek API call failed: {str(e)}")

    def invoke(self, messages: List[HumanMessage]) -> AIMessage:
        try:
            # Add user message to conversation
            self.messages.append({
                "role": "user",
                "content": messages[0].content
            })

            # Make API call exactly as in documentation
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages
            )

            # Add assistant's response to conversation history
            self.messages.append(response.choices[0].message)

            # Update token usage if available
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens_used += response.usage.total_tokens

            return AIMessage(content=response.choices[0].message.content)

        except Exception as e:
            print(f"DeepSeek API Error: {str(e)}")
            raise

    def get_token_usage(self) -> int:
        """Return total tokens used"""
        return self.total_tokens_used