import getpass
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_classic import hub
from langchain_classic.chains import RetrievalQA, create_history_aware_retriever  # langchain.chains -> ModuleNotFoundError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
# pip install -> 위 패키지들을 설치해준다

# llm.py 페이지의 목적 : 기능을 분리하여 관리한다

# 기능 분리
## 1. dictionary_chain, qa_chain 분리
## 2. llm의 분리 : llm은 사용되는 곳이 많으므로 분리


def get_llm(model='gpt-4o'):   # 다른 모델을 활용할 수도 있으므로, 매개변수로 지정해준다
    llm = ChatOpenAI(model=model)
    return llm
 
 

def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전 : {dictionary}
        사용자의 질문 : {{question}}
                                            
    """)
    dictionary_chain = prompt|llm|StrOutputParser()  

    return dictionary_chain


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')  # 디폴트 모델 : text-embedding-ada-002 -> 변경할 것
    index_name = "tax-markdown-index" 

    # from_existing_index: 벡터스토어 로드 (기존 인덱스를 불러옴)
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={"k": 4})  # retriever는 데이터베이스에서 정보를 가져오는 역할
    return retriever


# def get_qa_chain():
#     prompt = hub.pull("rlm/rag-prompt")
    
#     llm = get_llm()
#     retriever = get_retriever()
#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         # retriever=vector_store.as_retriever(),  # 다른 벡터 데이터베이스에서 as_retriever를 활용할 수 있다
#         retriever=retriever,
#         chain_type_kwargs={"prompt" : prompt}
#     )
    
#     return qa_chain


def get_qa_chain():

    llm = get_llm()
    retriever = get_retriever()

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )  # 이 시스템 프롬프트를 활용해서, 아래의 새로운 프롬프트를 만든다 (contextualize_q_prompt)

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    return rag_chain



def get_ai_message(user_message):

    dictionary_chain = get_dictionary_chain()
    qa_chain = get_qa_chain()
    
    tax_chain = {"query" : dictionary_chain} | qa_chain  
    ai_message = tax_chain.invoke({"question" : user_message})  # 질문이 사용자 쿼리임

    return ai_message['result']
