# Dependencies
<ul>
<li>Langchain</li>
<li>PyPDF2</li>
<li>python-dotenv</li>
<li>openai</li>
<li>faiss-cpu</li>
<li>altair</li>
<li>tiktoken</li>
</ul>

# Creating a Virtual Environment

```yaml
python3 -m venv .venv
. .venv/bin/activate
```


# Installing the dependencies

```yaml
pip install -m requirements.py
```

# Create a .env file and add your own key
```yaml
OPENAI_API_KEY=
```


# To run the code
```yaml
streamlit run app.py
```
