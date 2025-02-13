import boto3
import json
from typing import Dict, List, Optional, Union
from io import BytesIO
import PyPDF2
import logging

class S3DocumentHandler:
    """
    Handles retrieval and processing of guidance documents from S3 bucket
    to provide context for the chatbot. Similar to the resume builder's
    document handling system but optimized for chatbot interactions.
    """
    
    def __init__(self, bucket_name: str = "guides-and-context-for-chatbot"):
        """
        Initialize the document handler with S3 client and bucket configuration.
        
        Args:
            bucket_name (str): Name of the S3 bucket containing guidance documents
        """
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.logger = logging.getLogger(__name__)
        
    def get_document_keys(self) -> List[str]:
        """
        Retrieve list of all document keys in the bucket.
        
        Returns:
            List[str]: List of document keys (filenames)
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            self.logger.error(f"Error listing objects in bucket: {e}")
            return []
            
    def read_pdf_document(self, key: str) -> str:
        """
        Read and extract text from a PDF document in S3.
        
        Args:
            key (str): S3 object key for the PDF
            
        Returns:
            str: Extracted text content
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            pdf_content = response['Body'].read()
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
                
            return '\n'.join(text)
            
        except Exception as e:
            self.logger.error(f"Error reading PDF {key}: {e}")
            return ""
            
    def read_text_document(self, key: str) -> str:
        """
        Read text content from a text file in S3.
        
        Args:
            key (str): S3 object key for the text file
            
        Returns:
            str: Text content
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read().decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error reading text file {key}: {e}")
            return ""
            
    def get_context_for_topic(self, topic: str) -> str:
        """
        Retrieve relevant context documents based on the topic.
        
        Args:
            topic (str): Topic to find context for
            
        Returns:
            str: Combined relevant context
        """
        documents = []
        topic_lower = topic.lower()
        
        for key in self.get_document_keys():
            # Skip if file doesn't match topic
            if topic_lower not in key.lower():
                continue
                
            if key.endswith('.pdf'):
                content = self.read_pdf_document(key)
            else:
                content = self.read_text_document(key)
                
            if content:
                documents.append(content)
                
        return '\n\n'.join(documents)
        
    def get_all_context(self) -> str:
        """
        Retrieve and combine all guidance documents.
        
        Returns:
            str: Combined context from all documents
        """
        documents = []
        
        for key in self.get_document_keys():
            if key.endswith('.pdf'):
                content = self.read_pdf_document(key)
            else:
                content = self.read_text_document(key)
                
            if content:
                documents.append(content)
                
        return '\n\n'.join(documents)
        
    def create_prompt_with_context(self, user_input: str) -> str:
        """
        Create a prompt that includes relevant context from guidance documents.
        
        Args:
            user_input (str): The user's input message
            
        Returns:
            str: Formatted prompt with context
        """
        # First try to get topic-specific context
        context = self.get_context_for_topic(user_input)
        
        # If no topic-specific context found, use all context
        if not context:
            context = self.get_all_context()
        
        # Format the prompt with context
        prompt = f"""
Based on the following guidance and context documents:

{context}

Please help the user with their question:
{user_input}

When responding:
1. Use the context provided to give accurate, relevant information
2. If the context doesn't fully address the question, acknowledge this and provide general guidance
3. Maintain a helpful, professional tone
4. Be concise while being thorough
"""
        return prompt