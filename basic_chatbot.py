import boto3
import json
from typing import List, Dict, Optional

class BedrockChatbot:
    """
    A basic chatbot implementation using AWS Bedrock's Claude model.
    Maintains conversation history and handles message interactions.
    """
    
    def __init__(self, model_id: str = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'):
        """
        Initialize the chatbot with AWS Bedrock client and conversation settings.
        
        Args:
            model_id (str): The Bedrock model identifier to use for chat
        """
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = model_id
        self.conversation_history: List[Dict[str, str]] = []
        
    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role (str): The role of the message sender ('user' or 'assistant')
            content (str): The content of the message
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_response(self, user_input: str) -> Optional[str]:
        """
        Get a response from the model for the user's input.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            Optional[str]: The model's response, or None if an error occurs
        """
        # Add user's message to history
        self.add_to_history("user", user_input)
        
        try:
            # Prepare the request body
            request_body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50000,
                "temperature": 0.7,  # Slightly higher temperature for more creative responses
                "messages": self.conversation_history
            })
            
            # Make the API call
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=request_body
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            content = response_body.get('content', [])
            
            if content and isinstance(content, list) and 'text' in content[0]:
                assistant_response = content[0]['text']
                # Add assistant's response to history
                self.add_to_history("assistant", assistant_response)
                return assistant_response
            else:
                print("Unexpected response format from the model.")
                return None
                
        except Exception as e:
            print(f"An error occurred while getting the response: {e}")
            return None

def main():
    """
    Main function to run the chatbot in a terminal interface.
    """
    print("Welcome to the AWS Bedrock Chatbot!")
    print("Type 'exit' to end the conversation.")
    print("-" * 50)
    
    chatbot = BedrockChatbot()
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        # Get and display response
        response = chatbot.get_response(user_input)
        if response:
            print("\nAssistant:", response)
        else:
            print("\nSorry, I encountered an error processing your message.")

if __name__ == "__main__":
    main()