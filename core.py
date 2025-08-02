import os, asyncio

from dotenv import load_dotenv


from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain import hub
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
import streamlit as st
import datetime, json, base64


def save_output(query, vectorstore):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    res = f"[{date}]\n"
    res += f"Query: {query}\n"

    retrieved_documents_with_scores = vectorstore.similarity_search_with_score(query)

    for doc, score in retrieved_documents_with_scores:
        res += f"Document ID: {doc.id}\n"
        res += f"Source: {doc.metadata["source"]}\n"
        res += f"Document: {doc.page_content}\n"
        res += f"Relevance Score: {score}\n"
        res += "-" * 50 + "\n"

    with open("./result/result.txt", "a") as f:
        f.write(res)
        f.write("\n\n")


def get_llm_output(query, chat_history):

    # Community Cloud
    gemini_api = st.secrets["GEMINI_API"]
    gemini_model = st.secrets["GEMINI_MODEL"]
    gemini_embed_model = st.secrets["GEMINI_EMBED_MODEL"]
    pinecone_index = st.secrets["INDEX_NAME"]
    pinecone_api = st.secrets["PINECONE_API_KEY"]

    # Local
    # gemini_api = os.getenv("GEMINI_API")
    # gemini_model = os.getenv("GEMINI_MODEL")
    # gemini_model = os.getenv("GEMINI_MODEL")
    # gemini_embed_model = os.getenv("GEMINI_EMBED_MODEL")
    # pinecone_index = os.getenv("INDEX_NAME")
    # pinecone_api = os.getenv("PINECONE_API_KEY")
    
    os.environ["PINECONE_API_KEY"] = pinecone_api

    asyncio.set_event_loop(asyncio.new_event_loop())
    embed = GoogleGenerativeAIEmbeddings(
        model=gemini_embed_model,
        google_api_key=gemini_api
    )

    new_vectorstore_pinecone = PineconeVectorStore(
        index_name=pinecone_index, embedding=embed
    )

    chat = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=gemini_api)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=new_vectorstore_pinecone.as_retriever(search_kwargs={"k": 5}),
        prompt=rephrase_prompt,
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    save_output(query, new_vectorstore_pinecone)
    return result