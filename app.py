from flask import Flask, jsonify, request,render_template
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

app = Flask(__name__)
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
llm=Ollama(model="phi3:3.8b",)
vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

prompt_template = PromptTemplate(
    template="You are an AI assistant. Answer the question below in English and limit your response to 500 words.\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["question"]
)

def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversation_chain

conversation_chain = get_conversation_chain(vector_store)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        question = request.json['question']
        prompted_question = prompt_template.format(question=question)
        response = conversation_chain({"question": prompted_question})
        
        return jsonify({"answer": response['chat_history'][-1].content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
