import streamlit as st
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI


# ---- GLOBAL VARIABLES ---- #
with open("openai-api-key-ml-platform.txt", "r") as f:
    openai_api_key = f.read().strip()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True)


# ---- FUNCTIONS ---- #
def reset_messages():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]


def get_pandas_dataframe_agent(df):
    return create_pandas_dataframe_agent(llm,
                                         df,
                                         verbose=True,
                                         agent_type=AgentType.OPENAI_FUNCTIONS,
                                         handle_parsing_errors=True,
                                         )


# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="Explainable AI", layout='wide')

# ---- TITLE ---- #
st.markdown("<h1 style='text-align: center;'> DPI - ML Platform </h1>", unsafe_allow_html=True)
st.write('---')


# ---- LLM Chat ---- #
st.subheader("LLM Chatbot")
if st.session_state.processed_df is not None:
    df_agent = get_pandas_dataframe_agent(st.session_state.processed_df)

    if prompt := st.chat_input(placeholder="What is this data about?"):
        # Limit prompt to 100 tokens
        num_tokens = len(prompt.split())
        if num_tokens > 100:
            st.warning(f"Please limit your prompt to 100 tokens. You have used {num_tokens} tokens.")
            st.stop()
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = df_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
    
    reset_chat = st.button("Reset Chat", on_click=reset_messages)