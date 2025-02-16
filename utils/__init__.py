import json
import math
import pandas as pd
from IPython.display import clear_output, display
import os
import re

def load_json(filename, pass_error:bool=False)->dict:
    """Carrega dados de um arquivo JSON."""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception as e:
        if pass_error:
            return {}
        else:
            raise e

def update_json(obj,filename)->bool:
    """Atualiza dados de um arquivo JSON."""
    try:
        with open(filename, 'w') as file:
            json.dump(obj, file, indent=4)
        return True
    except Exception as e:
        return False
    

def format_time(seconds: float) -> str:
    if seconds is None or math.isnan(seconds) or math.isinf(seconds):
        return "0.0"
    
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours:02}:{minutes:02}:{int(seconds % 60):02}"
    elif seconds >= 60:
        minutes = int(seconds // 60)
        return f"{minutes:02}:{int(seconds % 60):02}"
    else:
        return f"{seconds:.2f}"

def update_table(models: list[str], predict_path: str, total_questions: int):

    with open(predict_path) as f:
        predict_data = json.load(f)
    
    models_dict = {model: [] for model in models}
    
    for question in predict_data.values():
        models_dict[question["model"]].append(question)

    table_data = []
    total_completed = 0
    total_corrects = 0
    total_errors = 0
    total_nulls = 0
    total_exec_time = 0
    total_min_time = float("inf")
    total_max_time = 0

    for model_name, model_questions in models_dict.items():
        completed = len(model_questions)
        corrects = sum(q["correct"] for q in model_questions)
        null_responses = sum(1 for q in model_questions if q["answer"] is None)
        errors = completed - corrects - null_responses
        accuracy = corrects / max(1, completed)
        total_time = sum(q["time"] for q in model_questions)
        avg_time = total_time / max(1, completed)
        max_time = max((q["time"] for q in model_questions), default=0)
        min_time = min((q["time"] for q in model_questions), default=float("inf"))
        estimated_time_left = avg_time * (total_questions - completed) if completed > 0 else 0

        total_completed += completed
        total_corrects += corrects
        total_errors += errors
        total_nulls += null_responses
        total_exec_time += total_time
        total_min_time = min(total_min_time, min_time)
        total_max_time = max(total_max_time, max_time)

        table_data.append([
            model_name,
            f"{completed} ({((100 * completed) / total_questions):.0f}%)",
            corrects,
            null_responses,
            errors,
            f"{accuracy:.0%}",
            format_time(total_time),
            format_time(estimated_time_left),
            format_time(avg_time),
            format_time(max_time),
            format_time(min_time)
        ])

    table_data.append([
        "TOTAL",
         f"{total_completed} ({((100 * total_completed) / (total_questions * len(models))):.0f}%)",
         total_corrects,
         total_nulls,
         total_errors,
         f"{(total_corrects / max(1, total_completed)):.0%}",
         format_time(total_exec_time),
         format_time((total_exec_time / max(1, total_completed)) * (total_questions - total_completed) if total_completed > 0 else 0),
         format_time(total_exec_time / max(1, total_completed)),
         format_time(total_max_time),
         format_time(total_min_time)
         ])

    df = pd.DataFrame(table_data, columns=["Model", "Finsh", "OK", "Null", "Err", "Acc", "Ttot", "Tle", "Tavg", "Tmax", "Tmin"])

    clear_output(wait=True)
    display(df)
