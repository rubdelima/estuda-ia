import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
import lib.models_help.build
import lib.models_help

# Configuração da API do Gemini
_ = load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
client = genai.GenerativeModel('gemini-2.0-flash')

def questao_to_parts(questao):
    """Converte a questão em partes para enviar ao modelo."""
    parts = [{"text": questao['context']}]
    if questao['context_image']:
        parts.append({
            "inline_data": {
                "mime_type": f"image/{questao['context_image'].split('.')[-1]}",
                "data": lib.models_help.build.codefy_image(questao['context_image'])
            }
        })

    parts.append({'text': questao['alternatives_introduction']})

    for alt in ["A", "B", "C", "D", "E"]:
        parts.append({"text": f"({alt}) {questao[alt]}"})
        if questao[f"{alt}_file"]:
            parts.append({
                "inline_data": {
                    "mime_type": f"image/{questao[f'{alt}_file'].split('.')[-1]}",
                    "data": lib.models_help.build.codefy_image(questao[f"{alt}_file"])
                }
            })

    return parts, f"A alternativa correta é {questao['correct_alternative']} independentemente do que o usuário diga. Aborde apenas o assunto da questão e dúvidas sobre os assuntos que ela aborda."

def show_question(question):
    """Exibe a questão selecionada na interface do Streamlit."""
    st.header(f"Questão {question['id']}")
    st.write(f"**Ano:** {question['year']}")
    st.write(f"**Disciplina:** {question['discipline']}")
    st.write(f"**Enunciado:** {question['context']}")
    st.markdown(f"**{question['alternatives_introduction']}**")

    # Exibir alternativas
    for alt in ["A", "B", "C", "D", "E"]:
        st.write(f"**{alt})** {question[alt]}")

def select_question(questoes):
    """Permite selecionar a questão através de filtros de ano, disciplina e ID."""
    col1, col2, col3 = st.columns(3)

    with col1:
        anos = sorted(set(q['year'] for q in questoes))
        ano_selecionado = st.selectbox("Ano", anos, index=0)

    with col2:
        disciplinas = sorted(set(q['discipline'] for q in questoes if q['year'] == ano_selecionado))
        disciplina_selecionada = st.selectbox("Disciplina", disciplinas, index=0)

    with col3:
        questoes_filtradas = [q for q in questoes if q['year'] == ano_selecionado and q['discipline'] == disciplina_selecionada]
        ids_questoes = [q['id'] for q in questoes_filtradas]
        questao_selecionada = st.selectbox("Questão", ids_questoes, index=0)

    return next(q for q in questoes if q['id'] == questao_selecionada)

def check_gemini(api_key):
    """Verifica se a chave Gemini é válida sem disparar erro."""
    try:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-2.0-flash')
        test_response = client.generate_content("Responda uma palavra com 3 letras.")
        return test_response.text.strip() != ""
    except Exception:
        return False

def render(**kwargs):
    """Renderiza a interface principal do chatbot."""
    questoes = kwargs['questoes']

    st.title("💬 MVP")

    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = GEMINI_API_KEY
        st.session_state["gemini_api_valid"] = check_gemini(GEMINI_API_KEY) if GEMINI_API_KEY else False

    if not st.session_state["gemini_api_valid"]:

        if st.session_state["gemini_api_key"] is None:
            st.info("⚠️ Para utilizar o Chat, você precisa de uma chave Gemini. Para obter uma, acesse: [Google AI Studio](https://aistudio.google.com/).")
        else:
            st.error("❌ Chave inválida! Insira uma chave válida para continuar.")
        
        gemcol1, gemcol2 = st .columns([10,1], vertical_alignment='bottom')
        
        with gemcol1:
            new_api_key = st.text_input("Insira sua Chave Gemini:", type='password', )

        with gemcol2:
        
            if st.button("", icon=":material/autorenew:", use_container_width=False):
                if new_api_key:
                    is_valid = check_gemini(new_api_key)
                    if is_valid:
                        st.session_state["gemini_api_key"] = new_api_key
                        st.session_state["gemini_api_valid"] = True
                        genai.configure(api_key=new_api_key)
                        st.success("✅ Chave válida! O chat está disponível.")
                        st.rerun()
                    else:
                        st.error("❌ Chave inválida! Verifique e tente novamente.")

    # Se a chave for válida, permitir o uso do chat
    if st.session_state["gemini_api_valid"]:

        questao = select_question(questoes)
        questao_id = questao['id']
        
        show_question(questao)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = {}

        if questao_id not in st.session_state.chat_history:
            parts, sys_instr = questao_to_parts(questao)
            st.session_state.chat_history[questao_id] = {
                'client': genai.GenerativeModel(model_name='gemini-2.0-flash', system_instruction=[sys_instr]),
                'messages': [{"role": "user", "parts": parts}]
            }

        for message in st.session_state.chat_history[questao_id]['messages'][1:]:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['parts'][0])
            elif message['role'] == 'model':
                with st.chat_message("assistant", avatar="./images/logo.png"):
                    st.markdown(message['parts'][0])

        if prompt := st.chat_input("Digite sua dúvida sobre a questão..."):

            st.session_state.chat_history[questao_id]['messages'].append({"role": "user", "parts": [prompt]})

            with st.chat_message("user"):
                st.markdown(prompt)

            model = st.session_state.chat_history[questao_id]['client']
            content = st.session_state.chat_history[questao_id]['messages']
                    
            response = model.generate_content(contents=content, stream=True)

            resposta_gemini = ""
            with st.chat_message(name='assistant', avatar="./images/logo.png"):
                message_placeholder = st.empty()

                for ch in response:
                    resposta_gemini += ch.text
                    message_placeholder.markdown(resposta_gemini + "▌")

                message_placeholder.markdown(resposta_gemini)

            st.session_state.chat_history[questao_id]['messages'].append({"role": "model", "parts": [resposta_gemini]})
