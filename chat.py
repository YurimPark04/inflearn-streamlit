import streamlit as st    

from dotenv import load_dotenv   # <python-dotenv>
from llm import get_ai_response


# 1. 페이지 컨피그를 해준다
# streamlit 코드 수정 후, 새로 고침으로 수정사항을 반영할 수 있다
st.set_page_config(page_title="테스트 챗봇", page_icon="🤖")

st.title("🤖 Langchain 데모 챗봇") # 제목 
# streamlit 이 자체적으로 h1 으로 매핑됨

st.caption("데모테스트 입니다")


load_dotenv() 

# 세션 설정 :  입력된 내용들을 session state에 저장해야 함
# session state : 한 세션 안에 유지되는 정보 (세션이 살아있는 동안에 유지되는 전역 변수)
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 채팅 인풋
# chat_message : name, avatar(default)
    # name: str | Literal['user', 'assistant', 'ai', 'human']


print(f"before == {st.session_state.message_list}")
# session state 에 있는 list에 들어가서 화면에 이전 데이터를 그려줘야 함

for message in st.session_state.message_list:  # 이전의 채팅 내용들을 그리고
    with st.chat_message(message["role"]): 
        st.write(message["content"])  


if user_question := st.chat_input(placeholder="궁금한 내용을 말해주세요"):  # 채팅을 입력할때마다 실행
    # pass

    # 사용자 질문
    with st.chat_message("user"): 
        st.write(user_question)  # 채팅을 입력할때마다, 전체 코드가 처음부터 실행되는 문제가 생김 (line by line으로 확인)
    st.session_state.message_list.append({"role": "user", "content": user_question}) # user 질문 저장

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)

        # AI 답변
        with st.chat_message("ai"): 
            # write_stream 으로 변경해야 Generator를 받게 됨
            ai_message = st.write_stream(ai_response)  # 채팅을 입력할때마다, 전체 코드가 처음부터 실행되는 문제가 생김 (line by line으로 확인)
            # string을 리턴 -> 최종으로 나온 전체 답변을 넣어줘야 다음 채팅이 들어올 때 에러가 안난다 (채팅할때마다 ui자동으로 그려지는 문제)
        st.session_state.message_list.append({"role": "ai", "content": ai_message}) # user 질문 저장


print(f"after === {st.session_state.message_list}")  # state 확인 