import streamlit as st

def table_tecnologias():
    st.markdown("## 🛠️ Tecnologias Utilizadas")

    tecnologias = [
        ("https://pandas.pydata.org/static/img/pandas.svg", "Pandas", 
         "Biblioteca de manipulação e análise de dados, essencial para o processamento das questões do ENEM."),
        ("https://numpy.org/images/logo.svg", "NumPy", 
         "Biblioteca fundamental para computação científica em Python, utilizada para operações matemáticas e manipulação de arrays."),
        ("https://images.seeklogo.com/logo-png/59/2/ollama-logo-png_seeklogo-593420.png", "Ollama", 
         "Plataforma de modelos de linguagem que utilizamos para avaliação e fine-tuning de modelos locais."),
        ("https://huggingface.co/front/assets/huggingface_logo.svg", "Hugging Face", 
         "Biblioteca que fornece ferramentas para processamento de linguagem natural, facilitando o uso e treinamento de modelos de IA."),
        ("https://matplotlib.org/_static/images/logo2.svg", "Matplotlib", 
         "Biblioteca de plotagem em Python, utilizada para visualização de dados e resultados das análises."),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Google_Gemini_logo.svg/1280px-Google_Gemini_logo.svg.png", "GenAI", 
         "Plataforma utilizada para integração com o modelo Gemini via API, permitindo avaliações de desempenho e viabilidade."),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/2560px-OpenAI_Logo.svg.png", "OpenAI", 
         "Biblioteca Python utilizada para acessar modelos avançados de IA via API, auxiliando na geração e explicação de questões."),
        ("https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png", "Streamlit", 
         "Framework utilizado para o desenvolvimento da interface web interativa do Estuda-AI."),
    ]

    # Criando a estrutura da tabela
    for img_url, nome, descricao in tecnologias:
        col1, col2 = st.columns([1, 4])  # Ajuste das colunas para manter alinhamento
        
        with col1:
            st.image(img_url, use_container_width=True)
        
        with col2:
            st.markdown(f"**{nome}**: {descricao}")  # Formata o nome e a descrição

        st.divider()


bem_vindo_text = """
    Bem-vindo ao Estuda-AI, uma plataforma inteligente desenvolvida para auxiliar estudantes na geração, explicação e análise de questões do ENEM utilizando modelos de Inteligência Artificial.    
    🔹 **Gere questões automaticamente**  
    🔹 **Receba explicações detalhadas**  
    🔹 **Interaja com modelos de IA para esclarecer dúvidas**
    """



def render(**kwargs):
    c1,c2,c3 = st.columns([2,3,2])
    c2.image('./images/logo.png')
    
    st.markdown(bem_vindo_text)
    
    st.markdown("## 📌 Sobre o Projeto")

    st.markdown("""

    O **Estuda-AI** é um projeto desenvolvido por alunos da **Universidade Federal de Pernambuco (UFPE)** durante a disciplina **[IF1006] Tópicos Avançados em Sistemas de Informação 3 - Transformação Digital com IA**. Nosso principal objetivo é criar um ambiente inteligente para auxiliar na geração, explicação e análise de questões do ENEM, utilizando modelos de Inteligência Artificial (IA).
    
    ## **🔍 Etapas do Projeto**

    ### 1️⃣ **Coleta e Processamento de Dados**
       - Construímos um banco de dados com questões do ENEM e suas respectivas resoluções.
       - Aplicamos pré-processamento de texto, incluindo tokenização, limpeza e normalização.
    
    ### 2️⃣ **Avaliação Inicial dos Modelos Locais**
       - Testamos diferentes modelos de IA localmente, utilizando o **Ollama**, analisando desempenho, custo e viabilidade.
       - Nosso objetivo foi encontrar os melhores modelos para realizar o fine-tuning posteriormente.
    
    ### 3️⃣ **Avaliação de Modelos via API**
       - Testamos outros modelos por requisição via API, para ser a forma mais fácil de conseguir o acesso da plataforma

    ### 4️⃣ **Fine-tuning dos Modelos**

    ### 5️⃣ **Testes e Validação**

    ---
    
        """)

    table_tecnologias()
    
    st.markdown("""
    ## **👥 Equipe**
    
    - [Erlan Lira Soares Junior](https://github.com/erlanliraa) - elsj
    - [Felipe de Barros Moraes](https://github.com/FelipeMoraes03) - fbm3
    - [Guilherme Maciel de Melo](https://github.com/GuilhermeMaciel75) - gmm7
    - [Rubens Nascimento de Lima](https://github.com/rubdelima) - rnl2
    
    ---
    """)
