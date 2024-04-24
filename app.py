# Import streamlit for app dev
import streamlit as st

# Import transformer classes for generaiton
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# Import torch for datatype attributes 
import torch
# Import the prompt wrapper...but for llama index
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

Your goal is to provide answers relating to the FS system of the University of Agder.<</SYS>>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper 
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Create and dl embeddings instance  
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)

# Create new service context instance
settings = Settings  # Assuming Settings is exposed as a pre-initialized object
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

# Initialize session state for chat log
if 'chat_log' not in st.session_state:
    st.session_state['chat_log'] = []

# Function to handle the message input
def handle_message():
    # Use the query engine to get a response for the user's message
    response = query_engine.query(st.session_state.input)
    
    # Update the chat log with the user's message and the bot's response
    st.session_state['chat_log'].append(("You", st.session_state.input))
    st.session_state['chat_log'].append(("Bot", response))
    
# Chat input box
user_input = st.text_input('Input your message here', key="input", on_change=handle_message)

# Display chat log
st.write("Chat History:")
for speaker, message in st.session_state['chat_log']:
    st.write(f"{speaker}: {message}")

# Display the response object and source text if available
if 'response' in st.session_state:
    with st.expander('Response Object'):
        st.json(st.session_state.response)  # Assuming the response is JSON-like data

    with st.expander('Source Text'):
        # If the response includes a method to get the source text, use it
        # Otherwise, you may need to format the source text yourself
        source_text = st.session_state.response.get_formatted_sources() if hasattr(st.session_state.response, 'get_formatted_sources') else "Source text not available."
        st.write(source_text)