import os, asyncio
# from dotenv import load_dotenv

# load_dotenv()
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain import hub
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
import streamlit as st


def get_llm_output(query, chat_history):

    # Community Cloud
    gemini_api = st.secrets(["GEMINI_API"])
    gemini_model = st.secrets(["GEMINI_MODEL"])
    gemini_embed_model = st.secrets(["GEMINI_EMBED_MODEL"])
    pinecone_index = st.secrets(["PINECONE_INDEX"])
    pinecone_api = st.secrets(["PINECONE_API"])
    os.environ["PINECONE_API_KEY"] = pinecone_api
    google_application_credentials = st.secrets(["GOOGLE_APPLICATION_CREDENTIALS"])

    # Local
    # gemini_api = os.getenv("GEMINI_API")
    # gemini_model = os.getenv("GEMINI_MODEL")
    # gemini_model = os.getenv(["GEMINI_MODEL"])
    # gemini_embed_model = os.getenv(["GEMINI_EMBED_MODEL"])
    # pinecone_index = os.getenv(["PINECONE_INDEX"])
    # pinecone_api = os.getenv(["PINECONE_API"])
    # google_application_credentials = os.getenv(["GOOGLE_APPLICATION_CREDENTIALS"])

    # if asyncio.get_event_loop_policy().get_event_loop() is None:
    asyncio.set_event_loop(asyncio.new_event_loop())
    embed = GoogleGenerativeAIEmbeddings(
        model=gemini_embed_model,
        credentials=google_application_credentials,
    )

    # new_vectorstore_faiss = FAISS.load_local("./LangChainSolver/VectorDB", embed, allow_dangerous_deserialization=True)
    new_vectorstore_pinecone = PineconeVectorStore(
        index_name=pinecone_index, embedding=embed
    )

    chat = ChatGoogleGenerativeAI(
        model=gemini_model, google_api_key=gemini_api
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=new_vectorstore_pinecone.as_retriever(),
        prompt=rephrase_prompt,
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result