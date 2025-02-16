import ollama
import re
from utils import load_json, update_json, update_table
import time
from ollama_help.build import text_question, get_images

def extract_answer(texto):
    pattern = r'\([ABCDE]\)|\{[ABCDE]\}'
    occs = re.findall(pattern, texto)
    return occs[-1][1] if occs else None


def test_ollama_models(models, questions, model_type, predict_file, short_response=False):
    """Executa os modelos e atualiza a tabela dinamicamente no notebook."""
    
    # Carrega o arquivo das predições
    predict_data = load_json(predict_file, pass_error=True)
    
    for model in models:
        for question in questions:
            question_id = str(question["id"])
            
            if f"{question_id}-{model}" in predict_data:
                continue
            
            question_text = text_question(question)
            
            start_time = time.time_ns()
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
            update_table(models, predict_file, len(questions))
