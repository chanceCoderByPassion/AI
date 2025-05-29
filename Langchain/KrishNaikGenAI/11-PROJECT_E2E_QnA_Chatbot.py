import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="QnA Chatbot wth Groq"
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage,AIMessage

# Prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant,please respond to the user queries"),
    ("user","Question:{question}")
])


def generate_response(question,api_key,llm,temperature,max_tokens):
    
    output_parser = StrOutputParser()
    llm = ChatGroq(api_key=api_key,model=llm,temperature=temperature,max_tokens=max_tokens)
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer

# Title of the app
st.title("QnA chatbot with Groq")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq api key:",type="password")

# Dropdown to select various Groq models
llm=st.sidebar.selectbox("Select a Groq Model",["gemma2-9b-it","llama3-70b-8192","whisper-large-v3"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

#  Main Interface for user input

st.write(" Go ahead and ask any question:")
user_input = st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,tokens)
    st.write(response)
else:
    st.write("Please provide the query !!!")