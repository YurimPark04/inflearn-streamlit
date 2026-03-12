import getpass
import os

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_classic import hub
from langchain_classic.chains import RetrievalQA, create_history_aware_retriever  # langchain.chains -> ModuleNotFoundError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from config import answer_examples
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


# Chatting History를 얹어주는 방식
##1. 딕셔너리로 관리하는 방법 -> 종료되면 메모리가 날아감
store={}  # session_id를 키로, BaseChatMessageHistory 객체를 값으로 저장하는 딕셔너리
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
    
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    # 새로운 qa_chain 프롬프트####################
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
    ########################################
    # retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever



def get_rag_chain():

    llm = get_llm()

    ####### Few Shot 을 위한 프롬프트 추가
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),

        ]
    )

    few_shot_prompt =FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples  = answer_examples
    )
    #################################################

    history_aware_retriever = get_history_retriever()

    # chain ##################################################################
    # Answer question
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,   # chatting history 인것처럼 착각하게 하기 위함
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    ##################################################################

    # chat_history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, 
        get_session_history,  # 채팅 히스토리까지 포함된 history_aware_retriever를 활용해서 원하는 기능을 구현할 수 있음
        input_messages_key="input",
        history_messages_key="chat_history",
        # return answer -> config.py 파일도 answer로 통일
        output_messages_key="answer",).pick('answer')   # ['answer] 로 해도 되는데, streaming 할때 안좋다 (에러)
    
    
    return conversational_rag_chain



def get_ai_response(user_message):

    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    
    # tax_chain = {"query" : dictionary_chain} | rag_chain  
    tax_chain = {"input" : dictionary_chain} | rag_chain    # input을 input으로 넣어줘야 함

    ai_response = tax_chain.stream(   # [invoke] -> stream으로 변경 : Any -> Iterator[Any로 바뀐다.
        {
            "question" : user_message
        },
        config={
            "configurable" : {"session_id" : "abc123"}   # Missing Key 'session_id' in configurable -> expected keys are[session_id] 관련 ValueError
        },
        )  # 질문이 사용자 쿼리임

    # return ai_message['result']
    return ai_response   # or pick을 활용
