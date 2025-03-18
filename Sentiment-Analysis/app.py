import streamlit as st
from llm_function import polarity_scores_roberta

st.set_page_config(
        page_title="Sentiment Analysis",
)

st.title('Sentiment Analysis')
sentiment = st.chat_input('Type your sentence to analyse')

if sentiment:
    st.write('Sentiment analysis for:')
    st.write(sentiment)
    st.write(polarity_scores_roberta(sentiment))
