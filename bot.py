# chatbot_streamlit.py

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# Load API keys
load_dotenv()

# Sidebar - Model Selection
st.sidebar.title("🤖 AI Model Selection")
ai_input = st.sidebar.selectbox(
    "Select your AI model",
    ["Chat-GTP", "Deepseek AI", "Meta AI"]
)

# Map model names to repo IDs
if ai_input == "Chat-GTP":
    ai_input = 'openai/gpt-oss-120b'
elif ai_input == "Deepseek AI":
    ai_input = 'deepseek-ai/DeepSeek-V3-0324'
elif ai_input == "Meta AI":
    ai_input = 'meta-llama/Llama-3.1-8B-Instruct'

# Initialize LLM
llm = HuggingFaceEndpoint(repo_id=ai_input)
model = ChatHuggingFace(llm=llm)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful AI assistant.")]

st.title("💬 AI Chatbot")
st.write("Chat with your selected AI model in real time!")

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get AI response
    result = model.invoke(st.session_state.chat_history)

    # Add AI message
    st.session_state.chat_history.append(AIMessage(content=result.content))

# Show chat history (TOP → BOTTOM)
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)
