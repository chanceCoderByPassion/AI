import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.embeddings import SentenceTransformerEmbeddings 
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
import io
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")

#  Setup streamlit app

st.title("Conversational RAG with pdf uploads and chat history")
st.write("Upload pdfs and chat with the content")

#  Input Groq api key
api_key=st.text_input("Enter your groq api key:",type="password")

# Check if groq api key is provided

if api_key:
    llm=ChatGroq(api_key=api_key,model_name="Gemma2-9b-It")

    # Chat interface
    session_id = st.text_input("Session id",value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}

    documents = []
    uploaded_files = st.file_uploader("Choose a pdf file",type="pdf",accept_multiple_files=False)
    #  Process uploaded files
    # if uploaded_files:
    #     documents = []
    #     for uploaded_file in uploaded_files:
    #         temppdf = f"./temp.pdf"
    #         with open(temppdf,"wb") as file:
    #             file.write(io.BytesIO(uploaded_file).getvalue())
    #             # file_name = uploaded_file.name
    #         loader = PyPDFLoader(temppdf)
    #         docs=loader.load()
    #         documents.extend(docs)

        # Split and create embeddings for the document
    
    if uploaded_files is not None:
        docs = []
        reader = PdfReader(uploaded_files)
        i = 1
        for page in reader.pages:
            docs.append(Document(page_content=page.extract_text(), metadata={'page':i}))
            i += 1
        # loader = PyPDFLoader(docs)
        # docs2=loader.load()
        # documents.extend(docs2)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        splits= text_splitter.split_documents(docs)
        vector_store = Chroma.from_documents(splits,embeddings)
        retriever = vector_store.as_retriever()

        contextualoize_q_system_prompt = (
            "Given a chat history and latest user question"
            "which might reference context in the chat history "
            "formulate a standalone question which can be understood"
            "without the chat history.Do not answer the question,"
            "just reformulate it if needed and otherwise return it as it is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualoize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        # QA prompt

        system_prompt = (
            "you are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer "
            "the question.if you dont know the answer, say that you "
            "dont know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)


    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conv_rag_chain = RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history= get_session_history(session_id)
        response = conv_rag_chain.invoke(
            {"input":user_input},
            config={
                "configurable":{"session_id":session_id}
            }
        )

    st.write(st.session_state.store)
    st.write("Assistant:",response['answer'])
    st.write("Chat History:",session_history.messages)

else:
    st.warning("Please enter the groq api key !")

