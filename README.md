# Government Scheme Chatbot
This Chatbot analyses the government schemes relates to Buisness and Entrepreneurship and reply accordingly
# Dependencies
<ul>
<li>Langchain</li>
<li>PyPDF2</li>
<li>ollama</li>
<li>faiss-cpu</li>
<li>altair</li>
<li>tiktoken</li>
</ul>

# Creating a Virtual Environment
This will create a virtual environment which will not interfere with the local libraries on your device
```yaml
python3 -m venv .venv
. .venv/bin/activate
```

# Installing the phi3 llm model
```yaml
ollama pull phi3:3.8b
```

# Installing the dependencies
Installing the required dependencies in the virtual environment
```yaml
pip install -m requirements.py
```

# Creating the vector store
This will create a vector store so that you can access the app.py. Make sure to delete the pre existing vector store and create a new vector store.
```yaml
python vector_store_creator.py
```

# To run the code
This will run a flask server which will be at http://127.0.0.1:5000 and you can use it to access it in a browser as well as in an API environment.
```yaml
python app.py
```

# Using Postman for API calling 
Use the script below and access it in API environment
```yaml
http://127.0.0.1:5000/get_answer
{
"question": "Sample Question"
}
```
