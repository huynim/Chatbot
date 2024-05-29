import os
import json
import torch
import shutil
import streamlit as st
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage

st.set_page_config(
    page_title="FSH - Chatbot",
    page_icon="https://kappa.lol/_3Z91",
)

# Define variable to hold model weights naming 
name = "bineric/NorskGPT-Llama-7B-v0.1"

@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/')
    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/', 
                                                 torch_dtype=torch.float16, 
                                                 rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True) 
    return tokenizer, model

tokenizer, model = get_tokenizer_model()

# Create a HF LLM using the llama index wrapper 
llm = HuggingFaceLLM(context_window=3900,
                    max_new_tokens=350,
                    generate_kwargs={"temperature": 0.1, "do_sample": False},
                    model=model,
                    tokenizer=tokenizer)

# Create and download embeddings instance 
embeddings = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)

# Create new service context instance
settings = Settings
settings.chunk_size = 1024
settings.llm = llm
settings.embed_model = embeddings

# Function to store file list in local storage
def store_file_list(file_list):
    file_list_json = json.dumps(file_list)
    st.write(
        f"""
        <script>
        localStorage.setItem('file_list', '{file_list_json}');
        </script>
        """,
        unsafe_allow_html=True
    )

# Retrieve file list from local storage
file_list = None
if st.session_state.get("file_list_json"):
    file_list = json.loads(st.session_state.file_list_json)

# Check if we need to reload data
current_file_list = sorted([f.name for f in Path('./data').iterdir() if f.is_file()])

if file_list != current_file_list:
    shutil.rmtree("./storage", ignore_errors=True)
    store_file_list(current_file_list)
else:
    if file_list is not None:
        current_file_list = file_list

# Function to load data
def load_data():    
    PERSISTED_DIR = "./storage"
    if not os.path.exists(PERSISTED_DIR):
        with st.spinner(text="Laster inn dokumentene..."):
            reader = SimpleDirectoryReader(input_dir="./data")
            documents = reader.load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=PERSISTED_DIR)
            return index
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSISTED_DIR)
        index = load_index_from_storage(storage_context)
        return index

index = load_data()

# Setup index query engine using LLM 
chat_engine = index.as_query_engine()

# Create title and image
st.markdown('''
    <div style='display: flex; justify-content: center; align-items: center;'>
        <img src='https://i.nuuls.com/f_HOD.png' width='60' style='border-radius: 50%'>
        <h1 style='text-align: center; 
            color: white; font-family: Impact, Arial, Helvetica, sans-serif; 
            -webkit-text-stroke-width: 1.5px; 
            -webkit-text-stroke-color: #C8102E;'>
            FSH på Llama 2
        </h1>
    </div>
''', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hei, spør meg et spørsmål om Felles studentsystem!"}
    ]

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="context",
                                                        llm=llm,
                                                        system_prompt=(
                                                            "Always respond in the query's language. As an expert on the FS system at the University of Agder,"
                                                            " your primary role is to provide detailed answers based on the knowledgebase. Provide all"
                                                            " the instructions from the article body in a structural way so the user can follow it easily."
                                                            " Always answer short and precise. Be very specific and to the point."
                                                        ),
                                                        )

if prompt := st.chat_input("Skriv en melding"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Tenker..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
