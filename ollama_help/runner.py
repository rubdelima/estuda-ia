import ollama
import re
from utils import load_json, update_json, test_table
import time
from ollama_help.build import text_question, get_images, context_description_prompt, context_description_image, context_prompt, answer_description_image, questions_description, questions_options
from itertools import product
from IPython.display import clear_output, display

def extract_answer(texto):
    pattern = r'\([ABCDE]\)|\{[ABCDE]\}'
    occs = re.findall(pattern, texto)
    return occs[-1][1] if occs else None


def test_ollama_models(models, questions, predict_file, short_response=False):
    """Executa os modelos e atualiza a tabela dinamicamente no notebook."""
    
    test_result = {'ok' : [], 'error' : []}
    
    # Carrega o arquivo das predições
    predict_data = load_json(predict_file, pass_error=True)
    
    for model in models:
        for question in questions:
            question_id = str(question["id"])
            try:
                if f"{question_id}-{model}" in predict_data:
                    continue
                
                start_time = time.time_ns()

                question_text = text_question(question)

                response = ollama.generate(
                    model=model, 
                    prompt=question_text,
                    options={
                        'temperature': 0.0,
                        'max_tokens': 1 
                    }if short_response else None,
                    images= get_images(question)
                )

                exec_time = (time.time_ns() - start_time) / 10**9  

                answer = extract_answer(response.response)

                predict_data[f"{question_id}-{model}"] = {
                    "question": question_id,
                    "model": model,
                    "response": response.response,
                    "response_length" : len(response.response),
                    "answer": answer,
                    "correct": question["correct_alternative"] == answer,
                    "time": exec_time,
                }

                update_json(predict_data, predict_file)
                df = test_table(predict_file, len(questions),models)
                clear_output(wait=True)
                display(df)
                test_result['ok'].append(({'question' : question, 'model' : model}))
            except Exception as e:
                test_result['error'].append(({'question' : question, 'model' : model, 'error' : str(e)}))
                print(f"Error ao gerar resposta para a pergunta {question_id} do modelo {model}: {e}")
                continue
    
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
                }

                update_json(predict_data, predict_file)
                test_result['ok'].append({'question' : question_id, 'model_vision' : model_vision, 'model_text' : model_text})
            except Exception as e:
                test_result['error'].append({'question' : question_id, 'model_vision' : model_vision, 'model_text' : model_text, 'error' : str(e)})
            
            df = test_table(predict_file, len(questions), models_str)
            clear_output(wait=True)
            display(df)
    
    return test_result