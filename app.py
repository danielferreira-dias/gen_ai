import streamlit as st
import asyncio
from services.processing import ProcessingService
from services.pii import AzureLanguageService
from services.agent import Agent

# ---------- SETUP ----------
st.set_page_config(page_title="PII-Aware Chatbot", page_icon="🤖")

azure_service = AzureLanguageService()
agent = Agent(model_name="gpt-5-chat")
processing_Service = ProcessingService()

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- TITLE ----------
st.title("🤖 PII-Aware Chatbot with Azure AI")

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- USER INPUT ----------
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # Detect PII and tokenize
        data = azure_service.pii_recognition_example(user_input)
        process_data = processing_Service.tokenize_pii(data=data)
        tokenized_text = process_data.get('tokenized_text', user_input)
    except Exception as e:
        st.error(f"⚠️ Error during PII processing: {e}")
        tokenized_text = user_input

    # ---------- AGENT RESPONSE ----------
    # Run async agent call safely
    response = asyncio.run(agent.llm_response(tokenized_text))

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
