import streamlit as st
from recommender import SHLRecommender

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("SHL Assessment Recommendation Engine")

query = st.text_input("Enter Job Description or Query")

if query:
    recommender = SHLRecommender()
    results = recommender.recommend(query)
    st.write("### Top Matching Assessments")
    st.dataframe(results.drop(columns="score"))
