from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

# reading the text
pdf_docs = "Business & Entrepreneurship.pdf"
text = ""
pdf_reader = PdfReader(pdf_docs)
for i, page in enumerate(pdf_reader.pages):
    content = page.extract_text()
    if content:
        text += content

# chunking the text
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=300,
    length_function=len,
)
chunks = text_splitter.split_text(text)  # Changed from raw_text to text

# creating the vector store
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vector_store = FAISS.from_texts(chunks, embeddings)  # Changed from texts to chunks

# saving the vector store
save_directory = "vector_store"
vector_store.save_local(save_directory)

# loading the vector store
new_vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
