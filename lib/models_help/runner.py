# Externals Libs
import ollama
import re
import time
from itertools import product
from IPython.display import clear_output, display
import multiprocessing
import warnings
import tqdm
import os
from dotenv import load_dotenv, find_dotenv
from google import genai
import openai
import traceback
import random

# Local Libs
from lib.utils import (load_json, update_json, test_table, 
                   format_test_table, models_info,
                   load_predictions,gen_modelos_str)
from lib.models_help.build import text_question, get_images, context_description_prompt, context_description_image, context_prompt, answer_description_image, questions_description, questions_options


_ = load_dotenv(find_dotenv())

try:
    client_openai = openai.Client()
except:
    client_openai = None

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client_gemini = genai.Client(api_key=GEMINI_API_KEY)

def ollama_generate(model, prompt, images):
    response = ollama.generate(
            model=model,
            prompt=prompt,
            images=images
    )
    return response.response

def gemini_generate(model, prompt, images):
    if images is None:
        images = []
        
    response = client_gemini.models.generate_content(
        model=model,
        contents = [prompt, *images]
    )
    
    return response.text

def open_ai(model, prompt, images):
    mensagens = [
        {'role': 'system', 'content' : 'Você entende muito de ciências-humanas'},
        {'role' : 'user', 'content' : prompt},
        {'role' : 'assistant', 'content' : 'Responda apenas a alternativa correta dentro dos parênteses, ex: (A)'} # alterar para system o agente
    ]
    response = client_openai.chat.completions.create(
        model=model,
        messages=mensagens,
        max_tokens=10,
        temperature=0.7
    )
    return response.choices[0].message.content
    
def get_model_runner(model):
    if 'gemini' in model:
        return gemini_generate
    if 'gpt' in model:
        return open_ai
    return ollama_generate


def model_generate(queue, model, prompt, images):
    try:
        resposta = get_model_runner(model)(model,prompt,images)
        queue.put(resposta)
    except Exception as e:
        queue.put(e)

def send_text(model, prompt, images=None, timeout=None):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=model_generate, args=(queue, model, prompt, images))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate() 
        process.join()
        raise TimeoutError(f"A geração de resposta excedeu o tempo limite de {timeout} segundos.")

    resultado = queue.get()
    if isinstance(resultado, Exception):
        raise resultado

    return resultado


def extract_answer(texto):
    pattern = r'\([ABCDE]\)|\{[ABCDE]\}'
    occs = re.findall(pattern, texto)
    return occs[-1][1] if occs else None


def question_text_vision(model_vision, question, images, timeout):
    # Caso a questão possua imagem no contexto
    if question['type'] in ['context-image', 'full-image']:
        context_prompt_str = context_description_image(question) 
        image_response = send_text(
            model=model_vision, 
            prompt=context_prompt_str,
            images= [images.pop(0)],
            timeout = timeout
        )
        question_text = context_description_prompt(question, image_response.response)
    else:
        question_text = context_prompt(question, False)
    # Caso a questão possua imagem nas alternativas
    if question['type'] in ['answer-image', 'full-image']:
        descriptions_list = []
        for i, ans in enumerate(["A", "B", "C", "D", "E"], start=1):
            ans_prompt_str = answer_description_image(question, ans)
            ans_response = ollama.generate(
                model=model_vision, 
                prompt=ans_prompt_str,
                images= [images.pop(0)]
            )
            descriptions_list.append(ans_response.response)
        question_text += "\n" + questions_description(question, descriptions_list)
    else:
        question_text += questions_options(question)
    
    return question_text

def test_models(questions, primary_models, secundary_models=None, predict_file=None, timeout=None, shuffle=False):    
    questions_str = list(map(lambda x : str(x['id']), questions))
    
    # Dicionário de Resultados do Treinamento
    test_result = {'ok' : [], 'error' : []}
    
    # Dicionário de Predições
    predict_data = (
        load_json(predict_file, pass_error=True) if predict_file is not None
        else load_predictions(questions_str,primary_models, secundary_models)
    )
    
    table_models = gen_modelos_str(primary_models, secundary_models=secundary_models)
    
    df = format_test_table(test_table(questions=questions_str, models=table_models), len(questions))
    clear_output(wait=True)
    display(df)
    
    to_update = []

    for primary_model, secundary_model, question in product(primary_models, secundary_models if secundary_models else [None], questions):
        model_name = f"{secundary_model}+{primary_model}" if secundary_model else primary_model
        predict_name = f"{question['id']}-{model_name}"
        if predict_name not in predict_data:
            to_update.append((primary_model, secundary_model, question, model_name, predict_name))

    if shuffle:
        random.shuffle(to_update)
    
    for primary_model, secundary_model, question, model_name, predict_name in tqdm.tqdm(to_update, desc="Teste"):
        question_id = question['id']
        try:          
            # Carrego as imagens, se houverem
            images = get_images(question)
            
            # Inicio o tempo da execução
            start_time = time.time_ns()
            
            # Cria o texto da questão para ser enviada
            question_text = text_question(question) if secundary_model is None else \
                question_text_vision(secundary_model, question, images, timeout=timeout//2)
            
            # Caso o modelo principal não seja de visão, poe como null as imagens para evitar problemas
            if (model := models_info.models.get(primary_model)) is None or model['algorithm'] != 'vision':
                images = None
                
            # Envia para a LLM principal
            response = send_text(
                model=primary_model, 
                prompt=question_text,
                images= images,
                timeout=timeout
            )
            
            # Encerra o tempo de execução do teste
            exec_time = (time.time_ns() - start_time) / 10**9 
            # Extrai a resposta
            answer = extract_answer(response)
            # Adiciona nos dados de predição
            predict_data[predict_name] = {
                "question": question_id,
                "model": model_name,
                "response": response,
                "response_length" : len(response),
                "answer": answer,
                "correct": question["correct_alternative"] == answer,
                "time": exec_time,
                "discipline" : question['discipline'],
                "timeout" : None
            }
            
            test_result['ok'].append(({'question' : question, 'model' : model}))
            
        except TimeoutError as te:
            predict_data[predict_name] = {
                "question": question_id,
                "model": primary_model if secundary_model is None else f"{secundary_model}+{primary_model}",
                "response": None,
                "response_length" : None,
                "answer": None,
                "correct": None,
                "time": None,
                "discipline" : question['discipline'],
                "timeout" : timeout
            }
            test_result['error'].append(({'question' : question['id'], 'model' : model, 'error' : str(te), 'traceback' : traceback.format_exc()}))
            error_str = f"Timeout Error, a questão {question_id} no modelo {model} passou de {timeout} segundos de execução"
            warnings.warn(error_str)
        except Exception as e:
            test_result['error'].append(({'question' : question['id'], 'model' : model, 'error' : str(e), 'traceback' : traceback.format_exc()}))
            error_str = f"Error ao gerar resposta para a pergunta {question_id} do modelo {model}: {e}"
            warnings.warn(error_str)
        finally:
            # Atualiza a tabela
            update_json(predict_data, "./data/predict_data/local_predictions.json" if predict_file is None else predict_file)
            
            # Plota a tabela 
            df = format_test_table(test_table(questions=questions_str, models=table_models), len(questions))
            clear_output(wait=True)
            display(df)
    
    return test_result