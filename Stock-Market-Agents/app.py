import streamlit as st
from agents import run_agent_team

st.set_page_config(
        page_title="Stock Market Agents"
)

st.title('Stock Market Agents')
companies = st.chat_input('Type companies to analyze')

if companies:
    st.write('Stock market analysis for:')
    st.write(companies)
    st.write(run_agent_team(companies))
