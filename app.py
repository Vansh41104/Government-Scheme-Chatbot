import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=300,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = Ollama(model="phi3.5")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory, 
    )
    return conversation_chain

def process_pdf_and_create_vectorstore():
    pdf_docs = "Business & Entrepreneurship.pdf"
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    
    # Save the vectorstore
    save_directory = "vector_store"
    vectorstore.save_local(save_directory)
    
    return vectorstore

def load_vectorstore():
    save_directory = "vector_store"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(save_directory):
        return FAISS.load_local(save_directory, embeddings, allow_dangerous_deserialization=True)
    else:
        return process_pdf_and_create_vectorstore()

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Conversation not initialized. Please try refreshing the page.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def initialize_conversation():
    vectorstore = load_vectorstore()
    conversation = get_conversation_chain(vectorstore)
    return conversation

def main():
    st.set_page_config(page_title="Government Scheme Chatbot", page_icon="🤖", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state or st.session_state.conversation is None:
        with st.spinner("Initializing the chatbot..."):
            st.session_state.conversation = initialize_conversation()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Government Scheme Chatbot 🤖")
    
    user_question = st.text_input("Ask a question about government schemes:")

    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()