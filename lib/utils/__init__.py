import json
import math
import pandas as pd
import os
from typing import Optional
from itertools import product
from lib.utils.models_info import models as models_json
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
    
    questions = list(map(str, questions))
    
    predictions = load_json('./data/predict_data/local_predictions.json')
    
    return {
        f"{question}-{model}": prediction for prediction in predictions.values()
        if ((model := prediction["model"]) in models) and ((question := str(prediction["question"])) in questions)
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

def test_table(questions: Optional[list[str]] = None, models: Optional[list[str]] = None, 
               predict_data: Optional[dict[str, dict]] = None, predict_path: Optional[str] = None):
    
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

    models_dict: dict[str, list] = {model: [] for model in models} if models is not None else {}

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

    table_data = []
    total_metrics = {
        "Finish": 0, "OK": 0, "Null": 0, "Err": 0, "Tout": 0, "Acc": 0,
        "Ttot": 0, "TTout": 0, "Tle": 0, "Tavg": 0, "Tmax": 0, "Tmin": float("inf"),
        "Size": 0
    }

    for model_name, model_questions in models_dict.items():
        metrics = {
            "Model": model_name,
            "Size": 0, 
            "Finish": len(model_questions),
            "OK": sum(1 for q in model_questions if q["correct"]),
            "Null": sum(1 for q in model_questions if q["answer"] is None and q["timeout"] is None),
            "Tout": sum(1 for q in model_questions if q.get("timeout") is not None),
            "Err": 0,  
            "Acc": 0,  
            "Prec": 0,  
            "Ttot": sum(q["time"] for q in model_questions if q['time']),
            "TTout": 0,  
            "Tle": 0, 
            "Tavg": 0, 
            "Tmax": max((q["time"] for q in model_questions if q['time']), default=0),
            "Tmin": min((q["time"] for q in model_questions if q['time']), default=float("inf")),
        }

        metrics["Err"] = metrics["Finish"] - metrics["OK"] - metrics["Null"] - metrics["Tout"]
        metrics["Acc"] = metrics["OK"] / max(1, metrics["Finish"])  # Evita divisão por zero
        metrics["Prec"] = metrics["OK"] / max(1, metrics["OK"] + metrics["Err"])
        metrics["TTout"] = metrics["Ttot"] + sum(q.get("timeout") for q in model_questions if q.get("timeout") is not None)
        metrics["Tle"] = (metrics["Ttot"] / max(1, metrics["Finish"])) * (total_questions - metrics["Finish"]) if metrics["Finish"] > 0 else 0
        metrics["Tavg"] = metrics["Ttot"] / max(1, metrics["Finish"])  # Tempo médio
        
        # Definição do tamanho do modelo
        if "+" in model_name:
            m1, m2 = model_name.split("+")
            model_size = models_json.get(m1, {}).get('size', 0) + models_json.get(m2, {}).get('size', 0)
            if model_size == 0:
                model_size = None
                warnings.warn(f"O modelo {model_name} não foi encontrado")
                time.sleep(1)
        else:
            model_size = models_json.get(model_name, {}).get('size')

        metrics["Size"] = round(model_size, 1) if model_size is not None else None

        # Atualizando totais
        for key in total_metrics:
            if key in ["Tmin"]:
                total_metrics[key] = min(total_metrics[key], metrics[key])
            elif key in ["Tmax"]:
                total_metrics[key] = max(total_metrics[key], metrics[key])
            else:
                total_metrics[key] += metrics[key]

        table_data.append(metrics)

    # Adiciona a linha TOTAL
    total_metrics["Model"] = "TOTAL"
    total_metrics["Acc"] = total_metrics["OK"] / max(1, total_metrics["Finish"])
    total_metrics["Prec"] = total_metrics["OK"] / max(1, total_metrics["OK"] + total_metrics["Err"])
    total_metrics["Tavg"] = total_metrics["Ttot"] / max(1, total_metrics["Finish"])
    total_metrics["Tle"] = sum(l['Tle'] for l in table_data)
    total_metrics["Size"] = round(total_metrics["Size"], 1)

    df = pd.DataFrame(sorted(table_data, key=lambda x: x["Acc"], reverse=True))  # Ordena antes de adicionar TOTAL
    df.loc[len(df)] = total_metrics
    
    return df


def format_test_table(df: pd.DataFrame, total_questions: Optional[int] = None) -> pd.DataFrame:
    """Converte valores numéricos de tempo para string formatada na tabela."""
    df_copy = df.copy()
    
    # Formatar colunas de tempo
    for col in ["Ttot", "Tle", "Tavg", "Tmax", "Tmin", "TTout"]:
        df_copy[col] = df_copy[col].apply(format_time)
    
    # Garantir que total_questions esteja correto
    if total_questions is None:
        total_questions = int(df["Finish"].max())  # Pegando o máximo da coluna, sem fatiamento

    # Aplicar formatação de porcentagem sem afetar a linha TOTAL
    df_copy["Finish"] = df_copy["Finish"].astype(object)
    df_copy.loc[df_copy.index[:-1], "Finish"] = df_copy.loc[df_copy.index[:-1], "Finish"].apply(
        lambda x: f"{x} ({100 * x / total_questions:.1f}%)"
    )

    # Ajustar a linha TOTAL separadamente
    finished_questions = int(df["Finish"].iloc[-1])
    df_copy.at[df_copy.index[-1], "Finish"] = f"{finished_questions} ({100 * finished_questions / (total_questions * (len(df_copy) - 1)):.1f}%)"

    return df_copy


def calcular_metricas(grupo):
    total = len(grupo)
    ok = grupo['correct'].sum()
    null = grupo['answer'].isna().sum()
    tout = grupo['timeout'].notna().sum()
    err = total - ok - null - tout
    ttot = grupo['time'].sum()
    ttout = grupo.loc[grupo['timeout'].notna(), 'time'].sum() + ttot
    tavg = grupo['time'].mean()
    tle = (total - (ok + null + tout + err))*tavg
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