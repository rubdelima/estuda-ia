import json
import math
import pandas as pd
import os
from typing import Optional
from itertools import product
from utils.models_info import models as models_json
import warnings
import time

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

def get_predict_data(models, questions):
    if questions and isinstance(questions[0], dict):
        questions = list(map(lambda x : str(x['id']), questions))
    
    predictions = load_json('./data/predict_data/local_predictions.json')
    
    return {
        f"{question}-{model}": prediction for prediction in predictions.values()
        if ((model := prediction["model"]) in models) and ((question := prediction["question"]) in questions)
    }
    

def gen_modelos_str(primary_models:list[str], questions=None, secundary_models=None):
    target = primary_models
    if secundary_models is not None:
        target = list(map(lambda x : f"{x[0]}+{x[1]}", product(secundary_models, primary_models)))
    
    if questions:
        target = list(map(lambda x : f"{x[0]}-{x[1]}", product(questions, target)))
    
    return target
    
def filter_predictions(models):
    all_predictions = load_json("./data/predict_data/local_predictions.json")
    
    return {
        pred_key : pred for pred_key, pred in all_predictions.items()
        if pred_key in models
    }
    
def load_predictions(questions, primary_models, secundary_models=None):
    if len(questions) > 0 and type(questions[0]) == dict:
        questions = [str(q["id"] )for q in questions]
    return filter_predictions(gen_modelos_str(primary_models, questions, secundary_models))
     
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

def test_table(questions:Optional[list[str]]=None, models:Optional[list[str]]=None, predict_data:Optional[dict[str, dict]]=None, predict_path:Optional[str]=None):
    if predict_data is None and (questions is None and models is None) and predict_path is None:
        raise ValueError("At least one of 'questions', 'predict_data', or 'predict_path' must be provided.")
    
    if questions is not None and len(questions) > 0:
        if isinstance(questions[0], dict):
            questions = [q["id"] for q in questions]
            
    if predict_data is None:
        if predict_path is not None:
            with open(predict_path, encoding='utf-8') as f:
               predict_data = json.load(f)
        
        else:
            predict_data = load_predictions(questions, models)
    
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
    total_completed = total_corrects = total_errors = total_nulls = total_size = total_exec_time = total_estimed_time_left = total_timeout = total_timeout_time = 0
    total_min_time = float("inf")
    total_max_time = 0
    
    for model_name, model_questions in models_dict.items():
        completed = len(model_questions)
        corrects = sum(1 for q in model_questions if q["correct"] )
        null_responses = sum(1 for q in model_questions if q["answer"] is None and q["timeout"] is not None)
        timeout = sum(1 for q in model_questions if q.get("timeout") is not None)
        timeout_time = sum(q.get("timeout")for q in model_questions if q.get("timeout") is not None)
        errors = completed - corrects - null_responses - timeout
        accuracy = corrects / max(1, completed)
        total_time = sum(q["time"] for q in model_questions if q['time'])
        avg_time = total_time / max(1, completed)
        max_time = max((q["time"] for q in model_questions if q['time']), default=0)
        min_time = min((q["time"] for q in model_questions if q['time']), default=float("inf"))
        estimated_time_left = (avg_time * (total_questions - completed)) if completed > 0 else 0
        
        if "+" in  model_name:
            m1, m2 = model_name.split("+")
            model_size = models_json.get(m1, {}).get('size', 0) + models_json.get(m2, {}).get('size', 0)
            if model_size == 0:
                model_size = None
                warnings.warn(f"O modelo {model_name} não foi encontrado")
                time.sleep(1)
        else:
            model_size = models_json.get(model_name, {}).get('size')
            

        total_completed += completed
        total_corrects += corrects
        total_nulls += null_responses
        total_timeout += timeout
        total_timeout_time += timeout_time
        total_errors += errors
        total_exec_time += total_time
        total_estimed_time_left += estimated_time_left
        total_min_time = min(total_min_time, min_time)
        total_max_time = max(total_max_time, max_time)
        total_size += model_size if model_size is not None else 0 

        table_data.append([
            model_name,( round(model_size,1) if model_size is not None else None), completed, corrects, null_responses, errors, timeout, accuracy,
            total_time, total_time+total_timeout_time, estimated_time_left, avg_time, max_time, min_time
        ])
    

    df = pd.DataFrame(table_data, columns=["Model", "Size", "Finish", "OK", "Null", "Err", "Tout", "Acc", "Ttot", "TTout","Tle", "Tavg", "Tmax", "Tmin"])
    
    df = df.sort_values("Acc", ignore_index=True, ascending=False)
    
    df.loc[len(df)] = [
        "TOTAL",round(total_size, 1), total_completed, total_corrects, total_nulls, total_errors, total_timeout,
        total_corrects / max(1, total_completed), total_exec_time, total_timeout_time,
        (total_exec_time / max(1, total_completed)) * (total_questions - total_completed) if total_completed > 0 else 0,
        total_exec_time / max(1, total_completed), total_max_time, total_min_time
    ]

    return df

def format_test_table(df:pd.DataFrame, total_questions:Optional[int]=None)->pd.DataFrame:
    """Converte valores numéricos de tempo para string formatada na tabela."""
    df_copy = df.copy()
    for col in ["Ttot", "Tle", "Tavg", "Tmax", "Tmin", "TTout"]:
        df_copy[col] = df_copy[col].apply(format_time)
    
    if total_questions is None:
        total_questions = int(df["Finish"][:-1].max())
    
    percentage_function = lambda x: f"{x} ({100*x/total_questions:.1f}%)"
    finished_questions = int(df["Finish"].iloc[-1])
    
    df_copy["Finish"] = df_copy["Finish"].apply(percentage_function)
    df_copy.at[df_copy.index[-1], "Finish"] = f"{finished_questions} ({100*finished_questions/(total_questions*(len(df_copy)-1)):.1f}%)"

    return df_copy

def calcular_metricas(grupo):
    total = len(grupo)
    ok = grupo['correct'].sum()
    null = grupo['answer'].isna().sum()
    err = total - ok - null
    tout = grupo['timeout'].notna().sum()
    ttot = grupo['time'].sum()
    ttout = grupo.loc[grupo['timeout'].notna(), 'time'].sum()
    tle = ttot + ttout
    tavg = grupo['time'].mean()
    tmax = grupo['time'].max()
    tmin = grupo['time'].min()
    return pd.Series({
        'Total': total,
        'OK': ok,
        'Null': null,
        'Err': err,
        'Tout': tout,
        'Ttot': ttot,
        'TTout': ttout,
        'Tle': tle,
        'Tavg': tavg,
        'Tmax': tmax,
        'Tmin': tmin
    })

def analisar_tabela(df, column):
    df['time'] = df['time'].fillna(0)

    resultado = df.groupby(column)\
        .apply(calcular_metricas).\
            reset_index()\
                .sort_values("OK", ascending=False, ignore_index=True)

    return resultado

def tabela_geral(questions, models):
    table = load_predictions(
    questions=questions,
    primary_models=models
    )
    return pd.DataFrame(table.values())