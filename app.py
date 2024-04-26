import streamlit as st
# Import transformer classes for generaiton
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# Import torch for datatype attributes 
import torch
# Import the prompt wrapper
from llama_index.core.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms.huggingface import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings.langchain import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
from llama_index.core import Settings
from llama_index.core import set_global_service_context
from llama_index.core import ServiceContext
# Import deps to load documents 
from llama_index.core import VectorStoreIndex, download_loader
from llama_index.core import SimpleDirectoryReader

# Define variable to hold llama2 weights naming 
name = "bineric/NorskGPT-Llama3-8b"
# Set auth token variable from hugging face 
auth_token = "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"

@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', token=auth_token)
    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , token=auth_token, torch_dtype=torch.float16, 
                            rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True) 
    return tokenizer, model
tokenizer, model = get_tokenizer_model()

# Create a system prompt 
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your goal is to provide answers relating to the FS system of the University of Agder. The FS system stands for Fellesstudentsystem.
As a designated expert on the FS system at the University of Agder, your main role is to provide detailed, technical answers to queries regarding the FS system.
Your responses should draw upon the established knowledgebase and compulsorily include the 'URL' of the user guidelines that matches the users question.
Do not create or make up links, but use the links that you are provided with in the database. Always assume the user is logged in to Service Now and the FS system.<</SYS>>
"""

# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper 
llm = HuggingFaceLLM(context_window=3900,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    generate_kwargs={"temperature": 0.1, "do_sample": False},
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Create and dl embeddings instance  
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
settings = Settings
settings.chunk_size = 1024
settings.llm = llm
settings.embed_model = embeddings

reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()

# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)
# Setup index query engine using LLM 
query_engine = index.as_query_engine()

# Create centered main title 
st.title('🐟 FSH')

# Initialize session state for chat_log if it does not exist
if 'chat_log' not in st.session_state:
    st.session_state['chat_log'] = []

# Function to handle the message input
def handle_message():
    user_input = st.session_state.input
    if user_input:  # Ensure input is not empty
        # Simulate response for illustrative purposes
        response = "This is a simulated response."
        
        # Update the chat log with the user's message and the bot's response
        st.session_state['chat_log'].append(("Deg", user_input))
        st.session_state['chat_log'].append(("FSH", response))
        
        # Clear the input field by setting the key's value to an empty string
        st.session_state.input = ""
        
        # Automatically scroll to the last message (simulate keeping the input at the bottom)
        st.experimental_rerun()

# Display chat log
for speaker, message in reversed(st.session_state['chat_log']):  # Display messages in reverse order
    st.text(f"{speaker}: {message}")

# Chat input box at the bottom
st.text_input('Hva kan jeg hjelpe deg med?', key="input", on_change=handle_message)

# Use an empty container to push the chat input to the bottom
st.empty()