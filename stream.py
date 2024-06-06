import streamlit as st
from transformers import pipeline

# Load NLP pipelines
sentiment_pipeline = pipeline('sentiment-analysis')
generation_pipeline = pipeline('text-generation')
translation_pipeline = pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
summarization_pipeline = pipeline('summarization')
NER_pipeline = pipeline("ner", grouped_entities=True)

# Streamlit app
st.title('NLP: Tasks Using Pipeline')

user_input = st.text_area('Enter text:')

task = st.selectbox('Choose a task:', ['Sentiment Analysis', 'Text Generation', 'Translation', 'Summarization', 'Named Entity Recognition'])

if st.button('Submit'):
    if task == 'Sentiment Analysis':
        result = sentiment_pipeline(user_input)
        st.write(f"Sentiment analysis: {result[0]['label']}, Score: {result[0]['score']}")
    elif task == 'Text Generation':
        result = generation_pipeline(user_input)
        st.write(f"Generated Text: {result[0]['generated_text']}")
    elif task == 'Translation':
        result = translation_pipeline(user_input)
        st.write(f"Translation: {result[0]['translation_text']}")
    elif task == 'Summarization':
        result = summarization_pipeline(user_input)
        st.write(f"Summarization: {result[0]['summary_text']}")
    elif task == 'Named Entity Recognition':
        result = NER_pipeline(user_input)
        for entity in result:
            st.write(f"Entity: {entity['word']}, Group: {entity['entity_group']}, Score: {entity['score']}")

