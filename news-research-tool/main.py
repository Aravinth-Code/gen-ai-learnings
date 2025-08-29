import sys
import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool")
st.sidebar.title("News Article URL")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature = 0.9, max_tokens = 500)

filepath = "E:\Project Learnings\ML Models\FAISS_Store_News_OpenAI"
if process_url_clicked:
    # load
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading started.........")
    data = loader.load()

    # split
    text_spliter = RecursiveCharacterTextSplitter(
        separators= ["\n\n", "\n", ".", ","],
        chunk_size = 1000,
    )
    main_placeholder.text("Text Split started.........")

    docs = text_spliter.split_documents(data)

    # embeddings
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Embeddings started.........")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    time.sleep(3)
    vectorstore_openai.save_local(filepath)
    main_placeholder.text("Embeddings End.........")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(filepath):
        embeddings = OpenAIEmbeddings()

        vectorstore = FAISS.load_local(
            filepath,
            embeddings,
            allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs = True)
        st.header("Answer: ")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            source_list = sources.split("\n")
            for source in source_list:
                st.write(source)