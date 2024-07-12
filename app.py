import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Reads the text from each PDF file and concatenates into a single string
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the text into smaller chunks to make it easier to process
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Convert text chunks into embeddings and store in a FAISS index for efficient similarity search
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the following question based on the provided context as accurately and comprehensively as possible. 
	If the answer is not found within the context, respond with "The answer is not available in the provided context." 
	Do not provide any information not included in the context.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")
    
    # Initialize session state for user information
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None

    # Function to collect user information
    def collect_user_info():
        with st.form("user_info_form"):
            name = st.text_input("Name")
            phone_number = st.text_input("Phone Number")
            email = st.text_input("Email")
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state.user_info = {
                    'name': name,
                    'phone_number': phone_number,
                    'email': email
                }

    # Check if user information is already collected
    if st.session_state.user_info is None:
        st.header("Please enter your details")
        collect_user_info()
    else:
        st.header(f"Welcome, {st.session_state.user_info['name']}")
        user_question = st.text_input("Ask a question from the PDF files")
        if user_question:
            user_input(user_question)

    # Optionally, display user information for confirmation
    if st.session_state.user_info is not None:
        st.sidebar.header("User Information")
        st.sidebar.write(f"Name: {st.session_state.user_info['name']}")
        st.sidebar.write(f"Phone Number: {st.session_state.user_info['phone_number']}")
        st.sidebar.write(f"Email: {st.session_state.user_info['email']}")
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
