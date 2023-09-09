#!streamlit run app_nisp.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1
import openai
import streamlit as st
import os

st.title("Local LLM Chatbot")

os.environ['OPENAI_API_KEY']="sk-111111111111111111111111111111111111111111111111"
os.environ['OPENAI_API_BASE']=os.environ["LLM_API_SERVER_BASE"]
#openai.api_key = os.environ['OPENAI_API_KEY']
#openai.api_base = os.environ['OPENAI_API_BASE']
model = "x"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "x"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me some questions!"):
    #context = """Use the following pieces of context to answer the users question.
    #If you don't know the answer, just say that you don't know, don't try to make up an answer.
    #
    #"""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            temperature=1.31,
            top_p=0.14,
            repetition_penalty=1.17,
            top_k=49,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})