import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text=""
    pdf_reader=PdfReader(pdf_docs)
    from typing_extensions import Concatenate
    for i, page in enumerate(pdf_reader.pages):
        content = page.extract_text()
        if content:
            text += content
    return text

def get_text_chunk(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=300,
        length_function=len,
    )
    chunks=text_splitter.split_text(raw_text)
    len(chunks)
    return chunks

def get_vectorstore(texts):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

def get_conversation_chain(store):
    llm=ChatOpenAI(model_name="gpt-3.5-turbo")
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store.as_retriever(),
        memory=memory, 
    )
    return conversation_chain

def handle_userinput(user_question):
    response=st.session_state.conversation({'question': user_question})
    st.session_state.chat_history=response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    with get_openai_callback() as cb:
        st.set_page_config(
            page_title="Government Scheme Chatbot", 
            page_icon="🤖", 
            layout="wide"
        )

        st.write(css, unsafe_allow_html=True)

        if "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        st.header("Government Scheme Chatbot")

        user_question=st.text_input("Hello I am your personalised bot which will tell you about the government schemes which you would like to know about")

        if user_question:
            handle_userinput(user_question)
       
        with st.sidebar:
            st.subheader("About")
            pdf_docs = "Business & Entrepreneurship.pdf"
            print("CREATING PDF")
            raw_text=get_pdf_text(pdf_docs)

            print("CREATING CHUNKS")
            text_chunk = get_text_chunk(raw_text)

            print("CREATING VECTOR STORE")
            vectorstore=get_vectorstore(text_chunk)
                    
            print("CREATING CONVERSATION CHAIN")
            st.session_state.conversation=get_conversation_chain(vectorstore)
            print(cb)



    


if __name__ == '__main__':
    main()