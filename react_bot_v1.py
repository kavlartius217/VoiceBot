# -*- coding: utf-8 -*-
"""1st December Reservation React Bot with Streamlit"""

import streamlit as st
from openai import OpenAI
from gtts import gTTS
import os
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool

# Page configuration
st.set_page_config(
    page_title="LeChateau Reservation Bot",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
        background-color: #FF4B4B;
        color: white;
    }
    .stAudio {
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .bot-message {
        background-color: #e8f0fe;
    }
    h1 {
        color: #1E1E1E;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid #FF4B4B;
    }
    .subheader {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-message {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success {
        background-color: #D4EDDA;
        color: #155724;
    }
    .error {
        background-color: #F8D7DA;
        color: #721C24;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Function to generate speech from text using gTTS
def text_to_speech(text):
    try:
        tts = gTTS(text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            return temp_audio.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

# Function to create agent executor with optimized settings
def create_agent_executor(llm, tools, prompt):
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        llm=llm,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=5,  # Reduced iterations
        early_stopping_method="generate",
        timeout=10  # 10 second timeout
    )

# Main UI
st.markdown("<h1>🍽️ LeChateau Reservation Bot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Welcome to our voice-enabled reservation system</p>", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎤 Voice Interface")
    st.markdown("Record your message below")
    
    # Use st.audio_input for recording
    audio_value = st.audio_input("Speak your request")

    if audio_value:
        st.markdown("<div class='status-message success'>Audio recorded successfully!</div>", unsafe_allow_html=True)
        st.audio(audio_value)
        
        # Create temporary file for Whisper API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_value.getvalue())
            audio_path = tmp_file.name

        # Transcribe using Whisper
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                customer_input = transcription.text
                st.markdown(f"<div class='chat-message user-message'>🗣️ You said: {customer_input}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            customer_input = None
        finally:
            # Clean up temp file
            os.remove(audio_path)

        if customer_input:
            # Initialize LLM and Embeddings
            llm = ChatOpenAI(
                model="gpt-4",
                api_key=st.secrets["OPENAI_API_KEY"]
            )
            embeddings = OpenAIEmbeddings(
                api_key=st.secrets["OPENAI_API_KEY"]
            )

            # Define tools
            def say_hello(customer_input):
                if any(greeting in customer_input.lower() for greeting in ["hello", "hey", "hi"]):
                    return "Hello! Welcome to LeChateu. How can I help you today?"
                return None

            tool1 = Tool.from_function(
                func=say_hello,
                name="say_hello_tool",
                description="use this tool to greet the customer after the customer has greeted you"
            )

            # Load CSV data
            csv_path = "table_data (1).csv"
            if not os.path.exists(csv_path):
                st.markdown("<div class='status-message error'>❌ Restaurant data not found!</div>", unsafe_allow_html=True)
                st.stop()

            try:
                csv = CSVLoader(csv_path)
                csv_data = csv.load()

                rcts1 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs2 = rcts1.split_documents(csv_data)

                db2 = FAISS.from_documents(docs2, embeddings)
                retriever2 = db2.as_retriever()

                tool2 = create_retriever_tool(
                    retriever2,
                    "reservation_data",
                    "csv file which has the reservation data"
                )

                tools = [tool1, tool2]

                # Updated prompt template
                prompt = PromptTemplate(
                    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
                    template='''You are an AI assistant managing reservations at LeChateau restaurant. Your primary tasks are:
1. Greet guests using the say_hello_tool when they greet you
2. Handle reservation requests using the reservation_data tool
3. Keep responses concise and direct

For reservations:
- Check available tables for the specified time and people count
- Show available tables with their locations
- Ask for table preference
- Confirm booking

{tools}

Use this format:
Question: {input}
Thought: [Brief thought about next action]
Action: [Tool name from {tool_names}]
Action Input: [Input value]
Observation: [Tool response]
Thought: [Brief thought about result]
Final Answer: [Concise response to guest]

Question: {input}
Thought:{agent_scratchpad}'''
                )

                # Process request with error handling
                try:
                    agent_exec = create_agent_executor(llm, tools, prompt)
                    with st.spinner("🤔 Processing your request..."):
                        try:
                            bot_response = agent_exec.invoke({
                                "input": customer_input,
                                "chat_history": st.session_state.get('chat_history', [])
                            })
                            response_text = bot_response.get('output', "I apologize, but I couldn't process your request. Could you please rephrase it?")
                        except TimeoutError:
                            response_text = "I apologize for the delay. Could you please simplify your request or break it into smaller parts?"
                        except Exception as e:
                            response_text = "I encountered an issue while processing your request. Please try again with a simpler query."
                            st.error(f"Processing error: {str(e)}")
                        
                        # Update chat history
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        
                        st.session_state.chat_history.append({
                            "user": customer_input,
                            "bot": response_text
                        })
                        
                        st.markdown(f"<div class='chat-message bot-message'>🤖 Bot: {response_text}</div>", unsafe_allow_html=True)

                        # Generate speech response
                        with st.spinner("🔊 Generating voice response..."):
                            speech_file = text_to_speech(response_text)
                            if speech_file:
                                st.audio(speech_file, format="audio/mp3")
                                os.unlink(speech_file)

                except Exception as e:
                    st.markdown(f"<div class='status-message error'>❌ System Error: {str(e)}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"<div class='status-message error'>❌ Error: {str(e)}</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 💬 Chat History")
    if st.session_state.get('chat_history'):
        for chat in st.session_state.chat_history:
            st.markdown(f"<div class='chat-message user-message'>👤 You: {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message bot-message'>🤖 Bot: {chat['bot']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("*No conversation history yet. Start by speaking into the microphone!*")

# Footer
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Made with ❤️ for LeChateau Restaurant</p>
        <p style='font-size: 0.8rem;'>Powered by OpenAI & Streamlit</p>
    </div>
""", unsafe_allow_html=True)
