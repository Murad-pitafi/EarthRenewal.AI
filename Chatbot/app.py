from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from dotenv import load_dotenv
import pyttsx3

load_dotenv()

app = Flask(__name__)

# Configure the Google Generative AI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from a PDF
def get_pdf_text():
    text = ""
    pdf_reader = PdfReader(r"C:\Users\Nizam\Desktop\Earthrenewal chatbot\Restoration_of_Degraded_Agricultural_Lan (1).pdf")
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split the text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate and save vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    return vector_store.as_retriever()

# Function to set up a conversational chain for Q&A using RetrievalQA
def get_conversational_chain(retriever: VectorStoreRetriever):
    prompt_template = """Answer the question as detailed as possible using the provided context.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",  
        retriever=retriever,  
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain

@app.route('/api/ask', methods=['POST'])
def ask_question():
    print("entered")
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400
    
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    chain = get_conversational_chain(retriever)
    
    response = chain.invoke({"query": user_question})
    result_text = response['result']
    
    return jsonify({'answer': result_text})
    print('exited')

if __name__ == '__main__':
    app.run(debug=True)
