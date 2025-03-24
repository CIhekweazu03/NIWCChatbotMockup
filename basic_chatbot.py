import boto3
import json
from typing import List, Dict, Optional
from io import BytesIO
import PyPDF2

class BedrockChatbot:
    """
    Enhanced chatbot implementation using AWS Bedrock's Claude model.
    Incorporates guidance documents from S3 for context-aware responses.
    """
    
    def __init__(
        self,
        model_id: str = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
    ):
        """
        Initialize the chatbot with AWS Bedrock client.
        
        Args:
            model_id (str): The Bedrock model identifier to use for chat
        """
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = model_id
        self.conversation_history: List[Dict[str, str]] = []
        self.has_sent_initial_context = False
        
        # Get all document content from S3
        s3 = boto3.client('s3')
        bucket_name = "guides-and-context-for-chatbot"
        
        try:
            response = s3.list_objects_v2(Bucket=bucket_name)
            self.guidance_info = []
            
            for obj in response.get('Contents', []):
                try:
                    key = obj['Key']
                    if key.lower().endswith('.pdf'):
                        # Handle PDF files
                        pdf_content = s3.get_object(Bucket=bucket_name, Key=key)['Body'].read()
                        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        self.guidance_info.append(text)
                    else:
                        # Handle text files
                        file_content = s3.get_object(Bucket=bucket_name, Key=key)['Body'].read().decode('utf-8')
                        self.guidance_info.append(file_content)
                except Exception as e:
                    print(f"Warning: Could not process file {key}: {e}")
                    continue
                    
            self.guidance_info = "\n\n".join(self.guidance_info)
        except Exception as e:
            print(f"Warning: Could not retrieve guidance documents: {e}")
            self.guidance_info = ""
        
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
        Get a context-aware response from the model for the user's input.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            Optional[str]: The model's response, or None if an error occurs
        """
        try:
            # For the first message, include the guidance information
            if not self.has_sent_initial_context:
                initial_prompt = f"""You are a helpful assistant with access to the following guidance information:

{self.guidance_info}

Please use this information when relevant to provide accurate and helpful responses. Be concise but thorough in your answers.

User's question: {user_input}"""
                self.add_to_history("user", initial_prompt)
                self.has_sent_initial_context = True
            else:
                # For subsequent messages, just add the user input normally
                self.add_to_history("user", user_input)
            
            # Prepare the request body
            request_body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50000,
                "temperature": 0.7,
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