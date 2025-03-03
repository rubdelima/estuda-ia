import streamlit as st

def render(**kwargs):
    st.title("📄 Fase 3")
    
    st.header("Introdução", anchor="introdução")
    st.write("Explicação introdutória sobre o tema...")

    st.header("Detalhes", anchor="detalhes")
    st.write("Aqui estão mais detalhes importantes...")

