import ollama
import re
from utils import (load_json, update_json, test_table, 
                   plots, format_test_table, models_info,
                   load_predictions, gen_modelos_str, new_test_table)
import time
from ollama_help.build import text_question, get_images, context_description_prompt, context_description_image, context_prompt, answer_description_image, questions_description, questions_options
from itertools import product
from IPython.display import clear_output, display
import pandas as pd
import multiprocessing
import ollama
import time

def ollama_generate(queue, model, prompt, images):
    try:
        resposta = ollama.generate(
            model=model,
            prompt=prompt,
            images=images
        )
        queue.put(resposta)
    except Exception as e:
        queue.put(e)

def send_text(model, prompt, images=None, timeout=None):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=ollama_generate, args=(queue, model, prompt, images))
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


def question_text_vision(model_vision, question, images):
    # Caso a questão possua imagem no contexto
    if question['type'] in ['context-image', 'full-image']:
        context_prompt_str = context_description_image(question) 
        image_response = ollama.generate(
            model=model_vision, 
            prompt=context_prompt_str,
            images= [images.pop(0)]
        )
        question_text = context_description_prompt(question, image_response.response)
    else:
        question_text = context_prompt(question, False)
    # Caso a questão possua imagem nas alternativas
    if question['type'] in ['context-image', 'full-image']:
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

def test_ollama_models(questions, primary_models, secundary_models=None, predict_file=None, timeout=None):    
    if primary_models is None and secundary_models is None:
        raise ValueError("Por favor, especifique pelo menos um modelo de texto ou uma visão.")
    
    questions_str = list(map(lambda x : x['id'], questions))
    
    # Dicionário de Resultados do Treinamento
    test_result = {'ok' : [], 'error' : []}
    
    # Dicionário de Predições
    predict_data = (
        load_json(predict_file, pass_error=True) if predict_file is not None
        else load_predictions(questions_str,primary_models, secundary_models)
    )
    
    models = list(product(primary_models, secundary_models if secundary_models else [None]))
    models_str = gen_modelos_str(questions_str, primary_models, secundary_models)
    
    for primary_model, secundary_model in models:
        for question in questions:
            question_id = str(question["id"])
            predict_name = (
                f"{question_id}-{primary_model}" if secundary_model is None
                else f"{question_id}-{secundary_model}+{primary_model}"
            )
            
            try:
                # Se o modelo já  estiver no dicionário
                if predict_name in predict_data:
                    old_timeout = predict_data[predict_name].get('timeout')
                    
                    # Se o valor  do timeout do anterior for None, significa que o treinamento aconteceu de forma bem sucedida
                    # Se houver um timeout, ele deve ser maior do que o timeout anterior
                    
                    if old_timeout is None or (timeout is not None and timeout <= old_timeout):
                        continue
                
                print(f"Realizando o Teste de : {predict_name}")
                time.sleep(10)
                
                # Carrego as imagens, se houverem
                images = get_images(question)
                
                # Inicio o tempo da execução
                start_time = time.time_ns()
                
                # Cria o texto da questão para ser enviada
                question_text = text_question(question) if secundary_model is None else \
                    question_text_vision(secundary_model, question, images)
                
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
                answer = extract_answer(response.response)

                # Adiciona nos dados de predição
                predict_data[predict_name] = {
                    "question": question_id,
                    "model": primary_model if secundary_model is None else f"{secundary_model}+{primary_model}",
                    "response": response.response,
                    "response_length" : len(response.response),
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
                raise Exception(te)
            
            except Exception as e:
                test_result['error'].append(({'question' : question, 'model' : model, 'error' : str(e)}))
                print(f"Error ao gerar resposta para a pergunta {question_id} do modelo {model}: {e}")
            
            finally:
                # Atualiza a tabela
                # update_json(predict_data, "./predict_data/local_predictions.json" if predict_file is None else predict_file)
                
                # Plota a tabela
                df = new_test_table(questions=questions_str, models=models, predict_data=predict_data)
                clear_output(wait=True)
                display(df)
    
    return test_result

def test_ollama_multi_models(text_models, vision_models, questions, predict_file):
    test_result = {'ok' : [], 'error' : []}
    predict_data = load_json(predict_file, pass_error=True)
    
    models = list(product(vision_models,text_models))
    models_str = list(map(lambda x : f"{x[0]}+{x[1]}", models))
    
    for model_vision, model_text in models:
        for question in questions:
            question_id = str(question["id"])
            try:
                if (test_id := f"{question_id}-{model_vision}+{model_text}") in predict_data:
                    continue
                
                start_time = time.time_ns()

                images = get_images(question)

                # Caso a questão possua imagem no contexto
                if question['type'] in ['context-image', 'full-image']:
                    context_prompt_str = context_description_image(question) 

                    image_response = ollama.generate(
                        model=model_vision, 
                        prompt=context_prompt_str,
                        images= [images.pop(0)]
                    )

                    question_text = context_description_prompt(question, image_response.response)
                else:
                    question_text = context_prompt(question, False)

                # Caso a questão possua imagem nas alternativas
                if question['type'] in ['context-image', 'full-image']:
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


                response = ollama.generate(
                    model=model_text,
                    prompt=question_text,
                )

                exec_time = (time.time_ns() - start_time) / 10**9

                answer = extract_answer(response.response)

                predict_data[test_id] = {
                    "question": question_id,
                    "model": f"{model_vision}+{model_text}",
                    "response": response.response,
                    "response_length" : len(response.response),
                    "answer": answer,
                    "correct": question["correct_alternative"] == answer,
                    "time": exec_time,
                    "discipline" : question['discipline']

                }

                update_json(predict_data, predict_file)
                test_result['ok'].append({'question' : question_id, 'model_vision' : model_vision, 'model_text' : model_text})
            except Exception as e:
                test_result['error'].append({'question' : question_id, 'model_vision' : model_vision, 'model_text' : model_text, 'error' : str(e)})
            
            df = test_table(predict_file, len(questions), models_str)
            clear_output(wait=True)
            display(df)
    
    return test_result