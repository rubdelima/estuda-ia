import json
import math
import pandas as pd
import os
from typing import Optional
from itertools import product

def load_json(filename, pass_error:bool=False)->dict:
    """Carrega dados de um arquivo JSON."""
    try:
        with open(filename, 'r', encoding='utf8') as file:
            return json.load(file)
    except Exception as e:
        if pass_error:
            return {}
        else:
            raise e

def update_json(obj,filename)->bool:
    """Atualiza dados de um arquivo JSON."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}

        data.update(obj)
        
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        return True
    except Exception as e:
        return False
    
def gen_modelos_str(questions, primary_models, secundary_models=None):
    target = primary_models
    if secundary_models is not None:
        target = map(lambda x : f"{x[0]}+{x[1]}", product(secundary_models, primary_models))
    
    return list(map(lambda x : f"{x[0]}-{x[1]}", product(questions, target)))
    
def filter_predictions(models):
    all_predictions = load_json("./predict_data/local_predictions.json")
    
    return {
        pred_key : pred for pred_key, pred in all_predictions.items()
        if pred_key in models
    }
    
def load_predictions(questions, primary_models, secundary_models=None):
    return filter_predictions(gen_modelos_str(questions, primary_models, secundary_models))
    
    
def format_time(seconds: float) -> str:
    if seconds is None or math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "ND"
    
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours:02}:{minutes:02}:{int(seconds % 60):02}"
    elif seconds >= 60:
        minutes = int(seconds // 60)
        return f"{minutes:02}:{int(seconds % 60):02}"
    else:
        return f"{seconds:.2f}"

# update_table
def test_table_old(predict_path: str, total_questions: int, models:list[str]|None=None)->pd.DataFrame:

    with open(predict_path) as f:
        predict_data = json.load(f)
    
    models_dict = {model : [] for model in models} if models is not None else {} #type: ignore
        
    for question in predict_data.values():
        if question["model"] in models_dict:
            models_dict[question["model"]].append(question)
        elif (models is not None) and (question["model"] not in models):
            continue
        else:
            models_dict[question["model"]] = [question]

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
         f"{total_completed} ({((100 * total_completed) / max(1, total_questions * len(models if models is not None else models_dict.keys()))):.0f}%)",
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

    return pd.DataFrame(table_data, columns=["Model", "Finsh", "OK", "Null", "Err", "Acc", "Ttot", "Tle", "Tavg", "Tmax", "Tmin"])

def new_test_table(questions:Optional[list[str]]=None, models:Optional[list[str]]=None, predict_data:Optional[dict[str, dict]]=None, predict_path:Optional[str]=None):
    if predict_data is None and (questions is None and models is None) and predict_path is None:
        raise ValueError("At least one of 'questions', 'predict_data', or 'predict_path' must be provided.")
    
    if predict_data is None:
        if predict_path is not None:
            with open(predict_path, encoding='utf-8') as f:
               predict_data = json.load(f)
        
        else:
            predict_data = filter_predictions(models)
    
    models_dict : dict[str, list] = {model: [] for model in models} if models is not None else {}
    
    for question in predict_data.values():
        if question["model"] in models_dict:
            models_dict[question["model"]].append(question)
        elif models is not None and question["model"] not in models:
            continue
        else:
            models_dict[question["model"]] = [question]
    
    total_questions = (
        len(questions) if questions else
        max(len(qs) for qs in models_dict.values()) if models_dict else len(predict_data)
    )
    
    table_data: list[list] = []
    total_completed = total_corrects = total_errors = total_nulls = total_exec_time = total_estimed_time_left = total_timeout = 0
    total_min_time = float("inf")
    total_max_time = 0
    
    for model_name, model_questions in models_dict.items():
        completed = len(model_questions)
        corrects = sum(q["correct"] for q in model_questions)
        null_responses = sum(1 for q in model_questions if q["answer"] is None)
        timeout = sum(1 for q in model_questions if q.get("timeout") is not None)
        errors = completed - corrects - null_responses - timeout
        accuracy = corrects / max(1, completed)
        total_time = sum(q["time"] for q in model_questions)
        avg_time = total_time / max(1, completed)
        max_time = max((q["time"] for q in model_questions), default=0)
        min_time = min((q["time"] for q in model_questions), default=float("inf"))
        estimated_time_left = (avg_time * (total_questions - completed)) if completed > 0 else 0

        total_completed += completed
        total_corrects += corrects
        total_nulls += null_responses
        total_timeout += timeout
        total_errors += errors
        total_exec_time += estimated_time_left
        total_min_time = min(total_min_time, min_time)
        total_max_time = max(total_max_time, max_time)

        table_data.append([
            model_name, completed, corrects, null_responses, errors, timeout, accuracy,
            total_time, estimated_time_left, avg_time, max_time, min_time
        ])
    
    table_data.append([
        "TOTAL", total_completed, total_corrects, total_nulls, total_errors, total_timeout,
        total_corrects / max(1, total_completed), total_exec_time,
        (total_exec_time / max(1, total_completed)) * (total_questions - total_completed) if total_completed > 0 else 0,
        total_exec_time / max(1, total_completed), total_max_time, total_min_time
    ])

    return pd.DataFrame(table_data, columns=["Model", "Finish", "OK", "Null", "Err", "Tout", "Acc", "Ttot", "Tle", "Tavg", "Tmax", "Tmin"])

def test_table(predict_path: str, total_questions: int | None = None, models: list[str] | None = None) -> pd.DataFrame:
    with open(predict_path) as f:
        predict_data = json.load(f)
    
    models_dict = {model: [] for model in models} if models is not None else {}
    
    for question in predict_data.values():
        if question["model"] in models_dict:
            models_dict[question["model"]].append(question)
        elif models is not None and question["model"] not in models:
            continue
        else:
            models_dict[question["model"]] = [question]
    
    if total_questions is None:
        total_questions = max(len(qs) for qs in models_dict.values()) if models_dict else len(predict_data)
    
    table_data = []
    total_completed = total_corrects = total_errors = total_nulls = total_exec_time = total_estimed_time_left = 0
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
        estimated_time_left = (avg_time * (total_questions - completed)) if completed > 0 else 0

        total_completed += completed
        total_corrects += corrects
        total_errors += errors
        total_nulls += null_responses
        total_exec_time += estimated_time_left
        total_min_time = min(total_min_time, min_time)
        total_max_time = max(total_max_time, max_time)

        table_data.append([
            model_name, completed, corrects, null_responses, errors, accuracy,
            total_time, estimated_time_left, avg_time, max_time, min_time
        ])
    
    table_data.append([
        "TOTAL", total_completed, total_corrects, total_nulls, total_errors,
        total_corrects / max(1, total_completed), total_exec_time,
        (total_exec_time / max(1, total_completed)) * (total_questions - total_completed) if total_completed > 0 else 0,
        total_exec_time / max(1, total_completed), total_max_time, total_min_time
    ])

    return pd.DataFrame(table_data, columns=["Model", "Finish", "OK", "Null", "Err", "Acc", "Ttot", "Tle", "Tavg", "Tmax", "Tmin"])

def format_test_table(df: pd.DataFrame) -> pd.DataFrame:
    """Converte valores numéricos de tempo para string formatada na tabela."""
    df_copy = df.copy()
    for col in ["Ttot", "Tle", "Tavg", "Tmax", "Tmin"]:
        df_copy[col] = df_copy[col].apply(format_time)
    df_copy["Finish"] = df_copy["Finish"].astype(str) + " (" + (df_copy["Finish"].astype(float) / df_copy["Finish"].max() * 100).round(0).astype(int).astype(str) + "%)"
    return df_copy
