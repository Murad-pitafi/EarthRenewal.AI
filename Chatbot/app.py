from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_pdf_text():
    text=""
    pdf_reader = PdfReader(r"C:\Users\Nizam\Desktop\Earthrenewal chatbot\Restoration_of_Degraded_Agricultural_Lan (1).pdf")
    for page in pdf_reader.pages:
        text+=page.extract_text()
    return text 
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks 

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain():
    prompt_template ="""Answer the questions as detailed as possible from the provided context
    context:\n{context}?\n
    Question:\n{question}\n
    Answer: 
    """
    model =ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,input_variables=['context','questions'])
    chain = load_qa_chain(model,chain_type='stuff',prompt=prompt)
    return chain 


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    #new_db = FAISS.load_local('faiss_index',embeddings)
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    #response = chain({'input_documents':docs,'question':user_question})
    response = chain.invoke({'input_documents': docs, 'question': user_question})

    print(response['output_text'])

def main():
    user_question='what is land restoration'
    user_input(user_question)
    #raw_text = get_pdf_text()
    #text_chunks = get_text_chunks(raw_text)
    #get_vector_store(text_chunks)
        
    
if __name__ =="__main__":
    main()
    
    
    
    
