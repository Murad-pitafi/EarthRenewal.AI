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
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure the Google Generative AI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from a PDF
def get_pdf_text():
    text = ""
    pdf_reader = PdfReader(r"D:\EarthRenewal.AI\Chatbot\CHatbot_corupus-earthrenewal_2.pdf")
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
 
def answer_without_context(question):
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.7)

    # Prepare the input as a string with instructions
    system_message = "You are an intelligent assistant. Answer the following question."
    user_message = question

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    try:
        # Call the model with the properly formatted input
        response = model(messages)
        return response.get('result', "Sorry, I don't know the answer to that.")
    except Exception as e:
        # Handle any exceptions and provide feedback
        return f"An error occurred: {str(e)}"


@app.route('/api/ask', methods=['POST'])
def ask_question():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Try to retrieve from the vector store first
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    chain = get_conversational_chain(retriever)
    
    # Get the response from the Retrieval chain
    response = chain.invoke({"query": user_question})
    result_text = response.get('result')
    print(result_text)
    # If the result mentions that it can't answer from the provided context, fallback to LLM
    if not result_text or "not" in result_text.lower():
        print("here")
        result_text = answer_without_context(user_question)
    
    return jsonify({'answer': result_text})

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, request, jsonify
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.vectorstores.base import VectorStoreRetriever
# from dotenv import load_dotenv
# from flask_cors import CORS

# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# # Configure the Google Generative AI API key
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Function to extract text from a PDF
# def get_pdf_text():
#     text = ""
#     pdf_reader = PdfReader(r"D:\EarthRenewal.AI\Chatbot\Restoration_of_Degraded_Agricultural_Lan (1).pdf")
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# # Function to split the text into manageable chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to generate and save vector store from text chunks
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")
    
#     return vector_store.as_retriever()

# # Function to set up a conversational chain for Q&A using RetrievalQA
# def get_conversational_chain(retriever: VectorStoreRetriever):
#     prompt_template = """Answer the question as detailed as possible using the provided context.
#     Context:\n{context}\n
#     Question:\n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
#     chain = RetrievalQA.from_chain_type(
#         llm=model,
#         chain_type="stuff",  
#         retriever=retriever,  
#         return_source_documents=False,
#         chain_type_kwargs={"prompt": prompt}
#     )
    
#     return chain

# # Fallback LLM direct answer function
# def answer_without_context(question):
#     model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.7)  # A bit higher temperature for creative responses
#     response = model({"query": question})
#     return response.get('result', "Sorry, I don't know the answer to that.")

# @app.route('/api/ask', methods=['POST'])
# def ask_question():
#     user_question = request.json.get('question')
#     if not user_question:
#         return jsonify({'error': 'No question provided'}), 400
    
#     # Try to retrieve from the vector store first
#     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
#     new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
#     retriever = new_db.as_retriever()
#     chain = get_conversational_chain(retriever)
    
#     # Get the response from the Retrieval chain
#     response = chain.invoke({"query": user_question})
#     result_text = response.get('result')
    
#     # If result_text is None or not sufficient, fall back to the model
#     if not result_text or "I don't know" in result_text:
#         print('not in corpus')
#         result_text = answer_without_context(user_question)
    
#     return jsonify({'answer': result_text})

# if __name__ == '__main__':
#     app.run(debug=True)


# # from flask import Flask, request, jsonify
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import RetrievalQA
# # from langchain.prompts import PromptTemplate
# # from langchain.vectorstores.base import VectorStoreRetriever
# # from dotenv import load_dotenv
# # import pyttsx3
# # from flask_cors import CORS

# # # Load environment variables from .env file
# # load_dotenv()

# # app = Flask(__name__)
# # CORS(app)

# # # Configure the Google Generative AI API key
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Function to extract text from a PDF
# # def get_pdf_text():
# #     text = ""
# #     pdf_reader = PdfReader(r"D:\EarthRenewal.AI\Chatbot\Restoration_of_Degraded_Agricultural_Lan (1).pdf")
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # # Function to split the text into manageable chunks
# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# #     chunks = text_splitter.split_text(text)
# #     return chunks

# # # Function to generate and save vector store from text chunks
# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
# #     # Ensure the directory exists
# #     save_path = r"D:\EarthRenewal.AI\Chatbot\faiss_index"
# #     os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

# #     # Save the FAISS index to the correct path
# #     vector_store.save_local(save_path)
    
# #     return vector_store.as_retriever()

# # # Function to set up a conversational chain for Q&A using RetrievalQA
# # def get_conversational_chain(retriever: VectorStoreRetriever):
# #     prompt_template = """Answer the question as detailed as possible using the provided context.
# #     Context:\n{context}\n
# #     Question:\n{question}\n
# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
# #     chain = RetrievalQA.from_chain_type(
# #         llm=model,
# #         chain_type="stuff",  
# #         retriever=retriever,  
# #         return_source_documents=False,
# #         chain_type_kwargs={"prompt": prompt}
# #     )
    
# #     return chain

# # @app.route('/api/ask', methods=['POST'])
# # def ask_question():
# #     user_question = request.json.get('question')
# #     if not user_question:
# #         return jsonify({'error': 'No question provided'}), 400
    
# #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# #     index_path = r"D:\EarthRenewal.AI\Chatbot\faiss_index"
    
# #     # Load the saved FAISS index from the correct directory
# #     new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
# #     retriever = new_db.as_retriever()
# #     chain = get_conversational_chain(retriever)
    
# #     response = chain.invoke({"query": user_question})
# #     result_text = response['result']
    
# #     return jsonify({'answer': result_text})

# # if __name__ == '__main__':
# #     # Preprocess the PDF to create vector store and start the app
# #     pdf_text = get_pdf_text()
# #     text_chunks = get_text_chunks(pdf_text)
# #     get_vector_store(text_chunks)  # Generate and save FAISS index
    
# #     app.run(debug=True)


# # from flask import Flask, request, jsonify
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import RetrievalQA
# # from langchain.prompts import PromptTemplate
# # from langchain.vectorstores.base import VectorStoreRetriever
# # from dotenv import load_dotenv
# # import pyttsx3
# # from flask_cors import CORS
# # load_dotenv()

# # app = Flask(__name__)
# # CORS(app)

# # # Configure the Google Generative AI API key
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Function to extract text from a PDF
# # def get_pdf_text():
# #     text = ""
# #     pdf_reader = PdfReader(r"D:\EarthRenewal.AI\Chatbot\Restoration_of_Degraded_Agricultural_Lan (1).pdf")
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # # Function to split the text into manageable chunks
# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# #     chunks = text_splitter.split_text(text)
# #     return chunks

# # # Function to generate and save vector store from text chunks
# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
# #     # Ensure the directory exists
# #     save_path = "faiss_index"
# #     os.makedirs(save_path, exist_ok=True)
    
# #     vector_store.save_local(save_path)
    
# #     return vector_store.as_retriever()

# # # Function to set up a conversational chain for Q&A using RetrievalQA
# # def get_conversational_chain(retriever: VectorStoreRetriever):
# #     prompt_template = """Answer the question as detailed as possible using the provided context.
# #     Context:\n{context}\n
# #     Question:\n{question}\n
# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
# #     chain = RetrievalQA.from_chain_type(
# #         llm=model,
# #         chain_type="stuff",  
# #         retriever=retriever,  
# #         return_source_documents=False,
# #         chain_type_kwargs={"prompt": prompt}
# #     )
    
# #     return chain

# # @app.route('/api/ask', methods=['POST'])
# # def ask_question():
# #     user_question = request.json.get('question')
# #     if not user_question:
# #         return jsonify({'error': 'No question provided'}), 400
    
# #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# #     index_path = r"D:\EarthRenewal.AI\Chatbot\faiss_index\index.faiss"
    
# #     # Load the saved FAISS index
# #     new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
# #     retriever = new_db.as_retriever()
# #     chain = get_conversational_chain(retriever)
    
# #     response = chain.invoke({"query": user_question})
# #     result_text = response['result']
    
# #     return jsonify({'answer': result_text})

# # if __name__ == '__main__':
# #     # Generate vector store and save it initially
# #     text = get_pdf_text()
# #     chunks = get_text_chunks(text)
# #     get_vector_store(chunks)
    
# #     # Run the Flask app
# #     app.run(debug=True)


# # from flask import Flask, request, jsonify
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import RetrievalQA
# # from langchain.prompts import PromptTemplate
# # from langchain.vectorstores.base import VectorStoreRetriever
# # from dotenv import load_dotenv
# # import pyttsx3
# # from flask_cors import CORS
# # from io import BytesIO
# # import wave
# # import speech_recognition as sr

# # load_dotenv()

# # app = Flask(__name__)
# # CORS(app)

# # # Configure the Google Generative AI API key
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Function to extract text from a PDF
# # def get_pdf_text():
# #     text = ""
# #     pdf_reader = PdfReader(r"D:\EarthRenewal.AI\Chatbot\Restoration_of_Degraded_Agricultural_Lan (1).pdf")
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # # Function to split the text into manageable chunks
# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# #     chunks = text_splitter.split_text(text)
# #     return chunks

# # # Function to generate and save vector store from text chunks
# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# #     vector_store.save_local("faiss_index")
    
# #     return vector_store.as_retriever()

# # # Function to set up a conversational chain for Q&A using RetrievalQA
# # def get_conversational_chain(retriever: VectorStoreRetriever):
# #     prompt_template = """Answer the question as detailed as possible using the provided context.
# #     Context:\n{context}\n
# #     Question:\n{question}\n
# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
# #     chain = RetrievalQA.from_chain_type(
# #         llm=model,
# #         chain_type="stuff",  
# #         retriever=retriever,  
# #         return_source_documents=False,
# #         chain_type_kwargs={"prompt": prompt}
# #     )
    
# #     return chain

# # # Speech-to-text function
# # def convert_audio_to_text(audio_data):
# #     recognizer = sr.Recognizer()
# #     audio_file = BytesIO(audio_data)
# #     with wave.open(audio_file, 'rb') as audio_wav:
# #         audio_data = sr.AudioFile(audio_wav)
# #         with audio_data as source:
# #             audio = recognizer.record(source)
# #         text = recognizer.recognize_google(audio)
# #     return text

# # @app.route('/api/ask', methods=['POST'])
# # def ask_question():
# #     # Check if audio file is sent
# #     if 'audio' in request.files:
# #         audio_file = request.files['audio']
# #         audio_data = audio_file.read()
# #         user_question = convert_audio_to_text(audio_data)
# #     else:
# #         user_question = request.json.get('question')

# #     if not user_question:
# #         return jsonify({'error': 'No question provided'}), 400

# #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# #     new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
# #     retriever = new_db.as_retriever()
# #     chain = get_conversational_chain(retriever)

# #     response = chain.invoke({"query": user_question})
# #     result_text = response['result']
    
# #     return jsonify({'answer': result_text})

# # if __name__ == '__main__':
# #     app.run(debug=True)



# # from flask import Flask, request, jsonify
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import RetrievalQA
# # from langchain.prompts import PromptTemplate
# # from langchain.vectorstores.base import VectorStoreRetriever
# # from dotenv import load_dotenv
# # import pyttsx3
# # from flask_cors import CORS
# # load_dotenv()

# # app = Flask(__name__)
# # CORS(app)

# # # Configure the Google Generative AI API key
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Function to extract text from a PDF
# # def get_pdf_text():
# #     text = ""
# #     pdf_reader = PdfReader(r"C:\Users\Nizam\Desktop\Earthrenewal chatbot\Restoration_of_Degraded_Agricultural_Lan (1).pdf")
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # # Function to split the text into manageable chunks
# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# #     chunks = text_splitter.split_text(text)
# #     return chunks

# # # Function to generate and save vector store from text chunks
# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# #     vector_store.save_local("faiss_index")
    
# #     return vector_store.as_retriever()

# # # Function to set up a conversational chain for Q&A using RetrievalQA
# # def get_conversational_chain(retriever: VectorStoreRetriever):
# #     prompt_template = """Answer the question as detailed as possible using the provided context.
# #     Context:\n{context}\n
# #     Question:\n{question}\n
# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
# #     chain = RetrievalQA.from_chain_type(
# #         llm=model,
# #         chain_type="stuff",  
# #         retriever=retriever,  
# #         return_source_documents=False,
# #         chain_type_kwargs={"prompt": prompt}
# #     )
    
# #     return chain

# # @app.route('/api/ask', methods=['POST'])
# # def ask_question():
# #     print("entered")
# #     user_question = request.json.get('question')
# #     if not user_question:
# #         return jsonify({'error': 'No question provided'}), 400
    
# #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# #     new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
# #     retriever = new_db.as_retriever()
# #     chain = get_conversational_chain(retriever)
    
# #     response = chain.invoke({"query": user_question})
# #     result_text = response['result']
    
# #     return jsonify({'answer': result_text})
# #     print('exited')

# # if __name__ == '__main__':
# #     app.run(debug=True)


# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import speech_recognition as sr
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import RetrievalQA
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv
# # import google.generativeai as genai
# # import os

# # # app = Flask(__name__)
# # # CORS(app)

# # # recognizer = sr.Recognizer()

# # # load_dotenv()

# # # # Configure the Google Generative AI API key
# # # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # def get_conversational_chain():
# # #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# # #     vector_store = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
# # #     retriever = vector_store.as_retriever()

# # #     prompt_template = """Answer the question as detailed as possible using the provided context.
# # #     Context:\n{context}\n
# # #     Question:\n{question}\n
# # #     Answer:
# # #     """
# # #     model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
# # #     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# # #     chain = RetrievalQA.from_chain_type(
# # #         llm=model,
# # #         chain_type="stuff",
# # #         retriever=retriever,
# # #         return_source_documents=False,
# # #         chain_type_kwargs={"prompt": prompt}
# # #     )

# # #     return chain

# # # @app.route('/api/ask', methods=['POST'])
# # # def ask():
# # #     data = request.get_json()
# # #     user_question = data.get('question')
# # #     if not user_question:
# # #         return jsonify({'error': 'No question provided'}), 400

# # #     chain = get_conversational_chain()
# # #     response = chain.invoke({"query": user_question})
# # #     answer = response['result']
    
# # #     return jsonify({'answer': answer})

# # # @app.route('/api/ask_audio', methods=['POST'])
# # # def ask_audio():
# # #     if 'audio' not in request.files:
# # #         return jsonify({'error': 'No audio file provided'}), 400

# # #     audio_file = request.files['audio']
# # #     with sr.AudioFile(audio_file) as source:
# # #         audio_data = recognizer.record(source)

# # #     try:
# # #         text = recognizer.recognize_google(audio_data)
# # #         chain = get_conversational_chain()
# # #         response = chain.invoke({"query": text})
# # #         answer = response['result']

# # #         return jsonify({'answer': answer})
# # #     except sr.UnknownValueError:
# # #         return jsonify({'error': 'Speech was not understood'}), 400
# # #     except sr.RequestError:
# # #         return jsonify({'error': 'API unavailable or network error'}), 500

# # # if __name__ == '__main__':
# # #     app.run(debug=True)



# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import joblib
# # import numpy as np
# # import os
# # import speech_recognition as sr
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import RetrievalQA
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv
# # import google.generativeai as genai
# # # import codecs
# # # # Initialize Flask app
# # # app = Flask(__name__)
# # # CORS(app)

# # # # Initialize the speech recognizer
# # # recognizer = sr.Recognizer()

# # # # Load environment variables
# # # load_dotenv()

# # # # Configure the Google Generative AI API key
# # # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # # Path to the TabNet Regressor model
# # # model_path = r'D:/EarthRenewal.AI/Model/Irrigation/tabnet_regressor.zip'

# # # with codecs.open(model_path, 'rb', encoding='utf-8', errors='ignore') as file:
# # #     tabnet_model = joblib.load(file)

# # # # Chatbot functionality
# # # def get_conversational_chain():
# # #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# # #     vector_store = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
# # #     retriever = vector_store.as_retriever()

# # #     prompt_template = """Answer the question as detailed as possible using the provided context.
# # #     Context:\n{context}\n
# # #     Question:\n{question}\n
# # #     Answer:
# # #     """
# # #     model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
# # #     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# # #     chain = RetrievalQA.from_chain_type(
# # #         llm=model,
# # #         chain_type="stuff",
# # #         retriever=retriever,
# # #         return_source_documents=False,
# # #         chain_type_kwargs={"prompt": prompt}
# # #     )

# # #     return chain

# # # # Route for text-based chatbot interaction
# # # @app.route('/api/ask', methods=['POST'])
# # # def ask():
# # #     data = request.get_json()
# # #     user_question = data.get('question')
# # #     if not user_question:
# # #         return jsonify({'error': 'No question provided'}), 400

# # #     chain = get_conversational_chain()
# # #     response = chain.invoke({"query": user_question})
# # #     answer = response['result']
    
# # #     return jsonify({'answer': answer})

# # # # Route for voice-based chatbot interaction
# # # @app.route('/api/ask_audio', methods=['POST'])
# # # def ask_audio():
# # #     if 'audio' not in request.files:
# # #         return jsonify({'error': 'No audio file provided'}), 400

# # #     audio_file = request.files['audio']
# # #     with sr.AudioFile(audio_file) as source:
# # #         audio_data = recognizer.record(source)

# # #     try:
# # #         text = recognizer.recognize_google(audio_data)
# # #         chain = get_conversational_chain()
# # #         response = chain.invoke({"query": text})
# # #         answer = response['result']

# # #         return jsonify({'answer': answer})
# # #     except sr.UnknownValueError:
# # #         return jsonify({'error': 'Speech was not understood'}), 400
# # #     except sr.RequestError:
# # #         return jsonify({'error': 'API unavailable or network error'}), 500

# # # # Irrigation functionality
# # # @app.route('/irrigation', methods=['POST'])
# # # def irrigation():
# # #     data = request.json
# # #     # Extract the input fields except 'DOY' and 'Active Root Zone\nPredicted'
# # #     features = [
# # #         'Precip\nmm', 'Irrig\nmm', '0 - 15\n%', '15 - 45\n%', '45 - 75\n%', 
# # #         '75 - 105\n%', '105 - 135\n%', '135 - 165\n%', '165 - 200\n%', 
# # #         'Growth\nStage', 'Root\nDepth', 'Canopy\nCover', 'Height\ncm', 
# # #         'N Applied\nkg/ha', 'ETr\nmm', 'Kcb', 'Ks', 'ETcb\nmm', 'Evap\nmm', 
# # #         'Deep Perc\nmm', 'ETc\nmm'
# # #     ]
# # #     input_data = np.array([data[feature] for feature in features]).reshape(1, -1)
    
# # #     # Make a prediction using the TabNet model
# # #     prediction = tabnet_model.predict(input_data)
# # #     return jsonify({"predicted_value": prediction[0]})

# # # # Run the Flask app
# # # if __name__ == '__main__':
# # #     app.run(debug=True)

# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import numpy as np
# # import os
# # import speech_recognition as sr
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import RetrievalQA
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv
# # import google.generativeai as genai
# # # from pytorch_tabnet.tab_model import TabNetRegressor

# # # Initialize Flask app
# # app = Flask(__name__)
# # CORS(app)

# # # Initialize the speech recognizer
# # recognizer = sr.Recognizer()

# # # Load environment variables
# # load_dotenv()

# # # Configure the Google Generative AI API key
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Path to the TabNet Regressor model
# # # model_path = r'D:\EarthRenewal.AI\Model\Irrigation\tabnet_regressor.zip'

# # # # Load the TabNet model
# # # loaded_model = TabNetRegressor()
# # # loaded_model.load_model(model_path)

# # # loaded_model = joblib.load('tabnet_model.h5')

# # # Chatbot functionality
# # def get_conversational_chain():
# #     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# #     vector_store = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
# #     retriever = vector_store.as_retriever()

# #     prompt_template = """Answer the question as detailed as possible using the provided context if nothing found in context answer the question from your knowledge.
# #     Context:\n{context}\n
# #     Question:\n{question}\n
# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# #     chain = RetrievalQA.from_chain_type(
# #         llm=model,
# #         chain_type="stuff",
# #         retriever=retriever,
# #         return_source_documents=False,
# #         chain_type_kwargs={"prompt": prompt}
# #     )

# #     return chain

# # # Route for text-based chatbot interaction
# # @app.route('/api/ask', methods=['POST'])
# # def ask():
# #     data = request.get_json()
# #     user_question = data.get('question')
# #     if not user_question:
# #         return jsonify({'error': 'No question provided'}), 400

# #     chain = get_conversational_chain()
# #     response = chain.invoke({"query": user_question})
# #     answer = response['result']
    
# #     return jsonify({'answer': answer})

# # # Route for voice-based chatbot interaction
# # @app.route('/api/ask_audio', methods=['POST'])
# # def ask_audio():
# #     if 'audio' not in request.files:
# #         return jsonify({'error': 'No audio file provided'}), 400

# #     audio_file = request.files['audio']
# #     with sr.AudioFile(audio_file) as source:
# #         audio_data = recognizer.record(source)

# #     try:
# #         text = recognizer.recognize_google(audio_data)
# #         chain = get_conversational_chain()
# #         response = chain.invoke({"query": text})
# #         answer = response['result']

# #         return jsonify({'answer': answer})
# #     except sr.UnknownValueError:
# #         return jsonify({'error': 'Speech was not understood'}), 400
# #     except sr.RequestError:
# #         return jsonify({'error': 'API unavailable or network error'}), 500

# # # # Irrigation functionality
# # # @app.route('/irrigation', methods=['POST'])
# # # def irrigation():
# # #     data = request.json
# # #     # Extract the input fields except 'DOY' and 'Active Root Zone\nPredicted'
# # #     features = [
# # #         'Precip\nmm', 'Irrig\nmm', '0 - 15\n%', '15 - 45\n%', '45 - 75\n%', 
# # #         '75 - 105\n%', '105 - 135\n%', '135 - 165\n%', '165 - 200\n%', 
# # #         'Growth\nStage', 'Root\nDepth', 'Canopy\nCover', 'Height\ncm', 
# # #         'N Applied\nkg/ha', 'ETr\nmm', 'Kcb', 'Ks', 'ETcb\nmm', 'Evap\nmm', 
# # #         'Deep Perc\nmm', 'ETc\nmm'
# # #     ]
# # #     input_data = np.array([data[feature] for feature in features]).reshape(1, -1)
    
# # #     # Make a prediction using the TabNet model
# # #     prediction = loaded_model.predict(input_data)
# # #     return jsonify({"predicted_value": prediction[0]})

# # # Run the Flask app
# # if __name__ == '__main__':
# #     app.run(debug=True)
