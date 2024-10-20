import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Get PDF content as text
def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# Split text into smaller chunks for vectorization
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=300,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

# Generate a vectorstore from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Create a conversational chain using the vectorstore and LLM
def get_conversation_chain(vectorstore):
    llm = Ollama(model="qwen2.5:1.5b")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

# Handle user input and display chat messages
def handle_userinput(user_question):
    if st.session_state.vectorstore is None:
        st.error("Vector store not initialized.")
        return
    
    # Process the user question as a new text chunk and update the vector store
    user_text_chunks = get_text_chunks(user_question)
    st.session_state.vectorstore = get_vectorstore(user_text_chunks)
    
    # Initialize a new conversation chain with updated vector store
    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
    
    # Get response from the conversation
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Initialize the vectorstore from the PDF
def initialize_vectorstore():
    pdf_docs = "Business & Entrepreneurship.pdf"
    
    if not pdf_docs:
        st.error(f"PDF file '{pdf_docs}' not found.")
        return None
    
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    
    # Create a vectorstore from the PDF text
    return get_vectorstore(text_chunks)

# Main function to initialize and run the chatbot
def main():
    st.set_page_config(page_title="Government Scheme Chatbot", page_icon="🤖", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        with st.spinner("Initializing vector store from PDF..."):
            st.session_state.vectorstore = initialize_vectorstore()

    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Government Scheme Chatbot 🤖")
    
    user_question = st.text_input("Ask a question about government schemes:")

    if user_question:
        handle_userinput(user_question)

# Run the Streamlit app
if __name__ == '__main__':
    main()
