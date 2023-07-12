import os 
import pickle
from apikey import apikey # import apikey from apikey folder, prompt user for their key when making this public 
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

os.environ['OPENAI_API_KEY'] = apikey

with st.sidebar:
    st.title('🤗 LLM Chat App')
    st.markdown('''
    ## About 
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchian.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                ''')
    
    add_vertical_space(5)
    st.write('Made with ❤️ by Douglas')

def main():
    st.header('Chat with pdf 💬')


    # Upload pdf file
    pdf = st.file_uploader('Upload your file here', type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        # used to retreive all text from pdf & concate to variable "text"
        text = " "
        for page in pdf_reader.pages:
            text += page.extract_text() 

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                # overlap with previous chunk, so that we can get context, or if they are interdependent 
                chunk_overlap = 200, 
                length_function = len
            )

            # used to break text attained from pdf (text) into chunks to be processed
        chunks = text_splitter.split_text(text=text)  

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write("Data read from local device")
        else:
            # embeddings (in charge of turning text into vectors for model to process)
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            # st.write("Embedding computation completed")

        # Accept user query
        query = st.text_input("Ask away 🏝️: ")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k = 3)

            llm = OpenAI(temperature=0.9, model='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)




if __name__ == "__main__":
    main()