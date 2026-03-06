import streamlit as st
from bot import RaceControlBot

st.set_page_config(page_title="FIA Race Control AI", page_icon="🏎️")
st.title("🏎️ FIA Race Control AI")
st.markdown("Ask any question about the 2026 F1 Sporting or Technical Regulations.")

@st.cache_resource
def get_bot():
    return RaceControlBot()

bot = get_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("E.g., What is the penalty for ignoring blue flags?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching FIA Regulations..."):
            response = bot.ask(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})