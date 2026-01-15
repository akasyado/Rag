import streamlit as st
from llm_backend import chain

if "message_log" not in st.session_state:
    st.session_state["message_log"] = []


for msg in st.session_state["message_log"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Type here")
if query:
    st.session_state["message_log"].append({"role" : "user" , "content" : query})
    with st.chat_message("user"):
        st.markdown(query)

    result = chain.invoke(query)
    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state["message_log"].append({"role" : "assistant" , "content" : result})




