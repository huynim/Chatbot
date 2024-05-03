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
name = "bineric/NorskGPT-Llama-7B-v0.1"
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

# Create a HF LLM using the llama index wrapper 
llm = HuggingFaceLLM(context_window=3900,
                    max_new_tokens=256,
                    generate_kwargs={"temperature": 0.1, "do_sample": False},
                    model=model,
                    tokenizer=tokenizer)

# Create and dl embeddings instance  
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
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
chat_engine = index.as_query_engine()

# Create centered main title 
col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image('https://i.nuuls.com/fsRNr.png', width=60)
with col2:
    st.title('FSH')

# Initialize chat history
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hei, spør meg et spørsmål om Felles studentsystem!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Laster inn kunnskapsbasen. Dette kan ta noen minutter."):
        reader = SimpleDirectoryReader(input_dir="./data")
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="context",
                                                            llm=llm,
                                                            system_prompt =(
                                                                "Always respond in the query's language. As an expert on the FS system at the University of Agder,"
                                                                " your primary role is to provide detailed answers based on the knowledgebase. Provide all"
                                                                " the instructions from the article body in a structural way so the user can follow it easily. Also ensure"
                                                                " that each response includes the exact URL from the 'URL' column of our knowledgebase for the"
                                                                " relevant UIA ServiceNow guidance referenced in the query, to avoid referencing incorrect or non-"
                                                                " existent links. If no link exists for the given guidance, do not provide a link."
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
