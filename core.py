from langchain_community.vectorstores import FAISS
import os, asyncio
from dotenv import load_dotenv
load_dotenv()
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain import hub
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore

def get_llm_output(query, chat_history):

    # if asyncio.get_event_loop_policy().get_event_loop() is None:
    asyncio.set_event_loop(asyncio.new_event_loop())
    embed = GoogleGenerativeAIEmbeddings(model = os.getenv("GEMINI_EMBED_MODEL"))
    
    # new_vectorstore_faiss = FAISS.load_local("./LangChainSolver/VectorDB", embed, allow_dangerous_deserialization=True)
    new_vectorstore_pinecone = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embed)

    chat = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), google_api_key=os.getenv("GEMINI_API"))

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=new_vectorstore_pinecone.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result