from app.backend.schemas import Questao, Chat, Model
from app.backend.models import filter_chat, load_question_context, add_message, send_question
from fastapi import FastAPI, WebSocket, HTTPException
from typing import Dict, List
from datetime import datetime
import asyncio  # Para streaming de respostas
import json


app = FastAPI()

# Carregando Questões
with open('./data/questoes/questoes.json') as f:
    questions_db = {
        questao['id'] : Questao(**questao)
        for questao in json.load(f)
    }

# Carregando Chats
with open('./data/chats.json') as f:
    chats_db = {
        chat['id'] : chat
        for chat in json.load(f)
    }

# Carregando modelos
models_db = [
    Model(name="phi4", image="/static/images/models/phi4.png"),
    Model(name="gemini-2.0-flash", image="/static/images/models/gemini-color.png"),
    Model(name="deepseek-r1", image="/static/images/models/deepseek-r1.png")
]

@app.get("/load_questions", response_model=List[Questao])
async def load_questions():
    return list(questions_db.values())


@app.get("/load_chats", response_model=List[Chat])
async def load_chats():
    return list(chats_db.values())


@app.get("/load_models", response_model=List[Model])
async def load_models():
    return models_db

@app.get("/load_chat/{chat_id}", response_model=List[dict])
async def load_chat(chat_id:str):
    chat = chats_db.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat não encontrado")
    
    return filter_chat(chat)
    

@app.post("/new_chat", response_model=Chat)
async def new_chat(model:str, question_id:str):
    questao = questions_db.get(question_id)
    if questao is None:
        raise HTTPException(status_code=404, detail="Questão não encontrada")
    
    messages = load_question_context(questions_db.get(question_id))
    
    chat = Chat(
        id=str(datetime.now().timestamp()),
        name= model + question_id,
        model=model,
        question=question_id,
        messages=messages
    )
    
    
    chats_db[chat.id] = chat
    
    return chat 
    
@app.websocket("/stream_chat/{chat_id}")
async def stream_chat(websocket: WebSocket, chat_id: str):
    await websocket.accept()

    chat = chats_db.get(chat_id)
    if chat is None:
        await websocket.send_text("Erro: Chat não encontrado.")
        await websocket.close()
        return

    model_name = chat.model

    try:
        for response_part in send_question(chat., model_name):
            await websocket.send_text(response_part['message']['content'])
            await asyncio.sleep(0.05)  # Simulando o tempo de resposta gradual da IA
    except WebSocketDisconnect:
        print(f"Conexão encerrada pelo cliente para chat {chat_id}")