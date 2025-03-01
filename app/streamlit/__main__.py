# Streamlit e Páginas
import streamlit as st
import app.streamlit.home as home
import app.streamlit.n01_dataset as n01
import app.streamlit.n02_local as n02
import app.streamlit.n03_online as n03
import app.streamlit.n04_fine as n04
import app.streamlit.mvp as mvp

import utils

questoes = utils.load_json('./data/questoes/questoes.json')
predicts = utils.load_json('./data/predict_data/local_predictions.json')

pages_dict = {
    'home' : home,
    'etapa1' : n01,
    'etapa2' : n02,
    'etapa3' : n03,
    'etapa4' : n04,
    'mvp' : mvp
}

# Realizando o setup da página
st.set_page_config(page_title="estuda-ai", layout="centered", page_icon='./images/logo.png')

if "pagina_selecionada" not in st.session_state:
    st.session_state["pagina_selecionada"] = "home"

# Logo do projeto
c1,c2,c3 = st.sidebar.columns([1,3,1])
c2.image("./images/logo.png", use_container_width=True)

# Página Inicial
st.sidebar.divider()
if st.sidebar.button("🏡 Página Inicial", key='home', use_container_width=True):
        st.session_state["pagina_selecionada"] = 'home'

# Etapas do Projeto
st.sidebar.markdown("### 📶 Etapas")

paginas = [
    ("📑 Coleta e Processamento dos Dados", "etapa1"),
    ("🖥️ Teste de Modelos Locais", "etapa2"),
    ("🛜 Teste de Modelos Online", 'etapa3'),
    ("🚀 Finetune dos Modelos", 'etapa4'),
]

for label, page in paginas:
    if st.sidebar.button(label, key=page,  use_container_width=True):
        st.session_state["pagina_selecionada"] = page

st.sidebar.divider()

if st.sidebar.button("💬 MVP", key='mvp', use_container_width=True, type='primary'):
        st.session_state["pagina_selecionada"] = 'mvp'


pagina_atual = st.session_state["pagina_selecionada"]

pages_dict.get(pagina_atual, home).render(predicts=predicts, questoes=questoes)