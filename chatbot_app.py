import streamlit as st
from typing import List, Dict
from basic_chatbot import BedrockChatbot

def initialize_chat_history() -> None:
    """
    Initialize the chat history in session state if it doesn't exist.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = BedrockChatbot()

def display_chat_messages() -> None:
    """
    Display all messages in the chat history.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def process_user_message(user_message: str) -> None:
    """
    Process a user message and get the chatbot's response with context indication.
    
    Args:
        user_message (str): The message from the user
    """
    # Add and display user message
    with st.chat_message("user"):
        st.markdown(user_message)
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and thinking..."):
            response = st.session_state.chatbot.get_response(user_message)
            if response:
                # Show context indicator
                st.caption("💡 Response includes guidance from our documentation")
                # Show response
                st.markdown(response)
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("I encountered an error processing your message. Please try again.")

def main():
    # Page configuration
    st.set_page_config(
        page_title="Chat Assistant",
        page_icon="💬",
        layout="centered"
    )
    
    # Initialize chat history
    initialize_chat_history()
    
    # Header
    st.title("💬 Chat Assistant")
    st.write("""
    Welcome! This chat assistant uses AWS Bedrock's Claude model to help answer your questions.
    Feel free to ask anything!
    """)
    
    # Display chat interface
    display_chat_messages()
    
    # Chat input
    if user_message := st.chat_input("What's on your mind?"):
        process_user_message(user_message)
    
    # Add a clear chat button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Add helpful information in the sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This chat assistant is powered by AWS Bedrock using the Claude model. 
        It maintains conversation history to provide contextual responses.
        
        Use the 'Clear Chat History' button above to start a fresh conversation.
        """)
        
        # Add some example questions
        st.header("Example Questions")
        st.write("""
        Try asking:
        - "Can you help me write a Python function?"
        - "What are the best practices for code documentation?"
        - "How do I optimize my code for performance?"
        """)

if __name__ == "__main__":
    main()