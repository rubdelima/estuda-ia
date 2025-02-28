import ollama
from models_help.build import get_messages, context_description_image, answer_description_image, get_images
from app.backend.schemas import Questao, Chat

def add_message(chat:Chat, role:str, message:str):
    chat.messages.append({ 'role' : role, 'content' : message})
    return chat

def load_question_context(question, secundary_model='minicpm-v')->list[dict]:
    descriptions = []
    
    images = get_images(question)
    
    if question['type'] in ['context-image', 'full-image']:
        response = ollama.generate(
            model=secundary_model,
            prompt= context_description_image(question),
            images= [images.pop(0)]
        )
        descriptions.append(response.response)
    
    if question['type'] in ['answer-image', 'full-image']:
        for i, ans in enumerate(["A", "B", "C", "D", "E"], start=1):
            ans_prompt_str = answer_description_image(question, ans)
            ans_response = ollama.generate(
                model=secundary_model, 
                prompt=ans_prompt_str,
                images= [images.pop(0)]
            )
            descriptions.append(ans_response.response)
    
    return get_messages(question, descriptions, images, '')

def filter_chat(chat:Chat):
    filter_messages = []
    for msg in chat.messages:
        if msg['role'] in ['user', 'assistant']:
            filter_messages.append(msg)
    return filter_messages

def send_question(messages, primary_model):
    response = ollama.chat(
        model=primary_model,
        messages=messages,
        stream=True,
    )
    
    return response