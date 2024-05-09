import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import Ollama, huggingface_hub

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

def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    # model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) 
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    # vector_store = FAISS.from_texts(texts, embeddings)

    # #Saving the vector store

    # save_directory="vector"
    # vector_store.save_local(save_directory)

    #Loading the vector store

    new_vector_store = FAISS.load_local("vector", embeddings, allow_dangerous_deserialization=True)
    return new_vector_store

def get_conversation_chain(store):
    llm=Ollama(model="llama3-gradient:1048k")
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

        st.header("Government Scheme Chatbot 🤖")

        user_question=st.text_input("Hello I am your personalised bot which will tell you about the government schemes which you would like to know about")

        if user_question:
            handle_userinput(user_question)

        pdf_docs = "Business & Entrepreneurship.pdf"
        # print("READING PDF")
        # raw_text=get_pdf_text(pdf_docs)

        # print("CREATING CHUNKS")
        # text_chunk = get_text_chunk(raw_text)

        print("CREATING VECTOR STORE")
        vectorstore=get_vectorstore()
        print(vectorstore)
                    
        print("CREATING CONVERSATION CHAIN")
        st.session_state.conversation=get_conversation_chain(vectorstore)
            
if __name__ == '__main__':
    main()