import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplate import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text=""
    for pdf_file in pdf_docs:
        pdf_reader=PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store=FAISS.from_texts(text_chunks,embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory=ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.markdown("# 📄🤖 DocQuill")
    st.write(css, unsafe_allow_html=True)  

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation_chain = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()