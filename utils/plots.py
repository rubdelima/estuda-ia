import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import format_time

def model_performance(df: pd.DataFrame):
    """Gera um gráfico de barras empilhadas mostrando a distribuição de acertos, nulos e erros por modelo."""
    df_sorted = df[df["Model"] != "TOTAL"].sort_values(by="OK", ascending=False)
    models = df_sorted["Model"]
    corrects = df_sorted["OK"]
    nulls = df_sorted["Null"]
    errors = df_sorted["Err"]
    total_questions = df_sorted["Finish"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(models, corrects, color='#99FF99', label='Acertos')
    bars2 = ax.bar(models, nulls, bottom=corrects, color='#C3C3C3', label='Nulos')
    bars3 = ax.bar(models, errors, bottom=corrects + nulls, color='#FF6666', label='Erros')
    
    for bar, total, correct, null, error in zip(bars1, total_questions, corrects, nulls, errors):
        ax.text(bar.get_x() + bar.get_width()/2, correct/2, f"{(correct/total*100):.1f}%", ha='center', va='center')
        ax.text(bar.get_x() + bar.get_width()/2, correct + null/2, f"{(null/total*100):.1f}%", ha='center', va='center')
        ax.text(bar.get_x() + bar.get_width()/2, correct + null + error/2, f"{(error/total*100):.1f}%", ha='center', va='center')
    
    ax.set_ylabel("Número de Questões")
    ax.set_xlabel("Modelos")
    ax.set_title("Desempenho dos Modelos")
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()

def time_metrics(df: pd.DataFrame):
    """Gera dois gráficos: um para tempo médio, mínimo e máximo por modelo, e outro para tempo total."""
    df_sorted = df[df["Model"] != "TOTAL"]
    models = df_sorted["Model"]
    avg_time = df_sorted["Tavg"]
    min_time = df_sorted["Tmin"]
    max_time = df_sorted["Tmax"]
    total_time = df_sorted["Ttot"] / 60  # Convertendo para minutos
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Primeiro gráfico - tempo médio, mínimo e máximo
    width = 0.2
    x = range(len(models))
    
    axes[0].bar(x, avg_time, width, label='Tempo Médio', color='#ffcc99')
    axes[0].bar([i + width for i in x], min_time, width, label='Tempo Mínimo', color='#99ccff')
    axes[0].bar([i + 2 * width for i in x], max_time, width, label='Tempo Máximo', color='#c2f0c2')
    
    for i, (avg, min_, max_) in enumerate(zip(avg_time, min_time, max_time)):
        axes[0].text(i, avg + 1, f"{avg:.2f}s", ha='center')
        axes[0].text(i + width, min_ + 1, f"{min_:.2f}s", ha='center')
        axes[0].text(i + 2 * width, max_ + 1, f"{max_:.2f}s", ha='center')
    
    axes[0].set_xticks([i + width for i in x])
    axes[0].set_xticklabels(models, rotation=45)
    axes[0].set_ylabel("Tempo (segundos)")
    axes[0].set_title("Métricas de Tempo por Modelo")
    axes[0].legend()
    
    # Segundo gráfico - tempo total por modelo
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2f0c2','#ffb3e6']
    bars = axes[1].bar(models, total_time, color=colors)
    
    for bar, time in zip(bars, total_time):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{time:.2f}m", ha='center')
    
    axes[1].set_ylabel("Tempo Total (minutos)")
    axes[1].set_title("Tempo Total por Modelo")
    
    plt.tight_layout()
    plt.show()

def accuracy_vs_time(df: pd.DataFrame):
    """Gera um gráfico de dispersão mostrando a correlação entre tempo médio e acurácia por modelo."""
    df_sorted = df[df["Model"] != "TOTAL"]
    models = df_sorted["Model"]
    avg_time = df_sorted["Tavg"]
    accuracy = df_sorted["Acc"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Paired(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        ax.scatter(avg_time.iloc[i], accuracy.iloc[i], color=colors[i], label=model)
        ax.text(avg_time.iloc[i], accuracy.iloc[i], model, fontsize=10, ha='right', color=colors[i])
    
    ax.set_xlabel("Tempo Médio (s)")
    ax.set_ylabel("Acurácia")
    ax.set_title("Correlação entre Tempo Médio e Acurácia por Modelo")
    
    plt.legend()
    plt.show()

def discipline_performance(df: pd.DataFrame, group_model: bool = False, normalize: bool = False):
    """Gera múltiplos gráficos de barras empilhadas mostrando a distribuição de acertos, nulos e erros por disciplina e modelo.
    Se group_model for True, agrupa os dados por modelo em vez de disciplina.
    Se normalize for True, normaliza os valores pelo total de questões em cada grupo.
    Limita a exibição a 4 modelos por gráfico para melhor visualização e adiciona separação entre grupos.
    """
    df_filtered = df.dropna(subset=["discipline"])
    
    group_by = "model" if group_model else "discipline"
    grouped = df_filtered.groupby([group_by, "model" if not group_model else "discipline"]).agg(
        OK=("correct", "sum"),
        Null=("answer", lambda x: x.isna().sum()),
        Total=("question", "count")
    ).reset_index()
    grouped["Err"] = grouped["Total"] - grouped["OK"] - grouped["Null"]
    grouped["Acc"] = grouped["OK"] / grouped["Total"]
    
    # Ordena os grupos pela acurácia
    sorted_groups = grouped.groupby(group_by)["Acc"].mean().sort_values(ascending=False).index
    grouped[group_by] = pd.Categorical(grouped[group_by], categories=sorted_groups, ordered=True)
    grouped = grouped.sort_values([group_by, "Acc"], ascending=[True, False])
    
    if normalize:
        grouped[["OK", "Null", "Err"]] = grouped[["OK", "Null", "Err"]].div(grouped["Total"], axis=0) * 100
    
    unique_groups = grouped[group_by].unique()
    group_chunks = [unique_groups[i:i+4] for i in range(0, len(unique_groups), 4)]
    
    for group_subset in group_chunks:
        fig, ax = plt.subplots(figsize=(max(12, len(group_subset) * 2), 8))
        width = 0.3
        colors = ['#99FF99', '#C3C3C3', '#FF6666']
        labels = ['Acertos', 'Nulos', 'Erros']
        
        tick_positions = []
        tick_labels = []
        x_index = 0
        spacing = 2
        
        for category in group_subset:
            category_data = grouped[grouped[group_by] == category]
            
            if category_data.empty:
                continue
            
            for i, (_, row) in enumerate(category_data.iterrows()):
                correct, null, error = row["OK"], row["Null"], row["Err"]
                total = row["Total"] if not normalize else 100
                
                tick_positions.append(x_index)
                tick_labels.append(f"{row['model'] if not group_model else row['discipline']}\n{category}")
                
                bottom = 0
                ax.bar(x_index, correct, width, color=colors[0], bottom=bottom, label=labels[0] if x_index == 0 else "")
                ax.text(x_index, bottom + correct / 2, f"{correct:.1f}%" if normalize else f"{int(correct)}", ha='center', va='center')
                bottom += correct
                
                ax.bar(x_index, null, width, color=colors[1], bottom=bottom, label=labels[1] if x_index == 0 else "")
                ax.text(x_index, bottom + null / 2, f"{null:.1f}%" if normalize else f"{int(null)}", ha='center', va='center')
                bottom += null
                
                ax.bar(x_index, error, width, color=colors[2], bottom=bottom, label=labels[2] if x_index == 0 else "")
                ax.text(x_index, bottom + error / 2, f"{error:.1f}%" if normalize else f"{int(error)}", ha='center', va='center')
                
                x_index += 1
            
            x_index += spacing  # Adiciona espaço entre grupos
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90)
        ax.set_ylabel("Proporção (%)" if normalize else "Número Total de Questões")
        ax.set_xlabel("Modelos e Disciplinas" if not group_model else "Disciplinas e Modelos")
        ax.set_title("Desempenho por Disciplina e Modelo" if not group_model else "Desempenho por Modelo e Disciplina")
        ax.legend()
        plt.show()
        
def discipline_time_performance(df: pd.DataFrame, group_model: bool = False):
    """Gera múltiplos gráficos de barras mostrando o tempo médio por disciplina e modelo.
    Se group_model for True, agrupa os dados por modelo e disciplina.
    O eixo Y representa o tempo médio em segundos.
    Limita a exibição a 2 disciplinas por gráfico se agrupado por disciplina e 3 disciplinas por gráfico se agrupado por modelo.
    Mantém agrupamentos juntos e adiciona legenda de cores.
    """
    df_filtered = df.dropna(subset=["discipline"]).copy()
    df_filtered["time"] = pd.to_numeric(df_filtered["time"], errors="coerce")  # Garante que os tempos sejam numéricos
    
    group_by = "model" if group_model else "discipline"
    grouped = df_filtered.groupby([group_by, "discipline" if group_model else "model"]).agg(
        Avg_Time=("time", "mean"),
        Count=("question", "count")
    ).reset_index()
    
    # Ordena os grupos pelo maior tempo médio
    sorted_groups = grouped.groupby(group_by)["Avg_Time"].mean().sort_values(ascending=False).index
    grouped[group_by] = pd.Categorical(grouped[group_by], categories=sorted_groups, ordered=True)
    grouped = grouped.sort_values([group_by, "Avg_Time"], ascending=[False, False])
    
    max_groups = 2 if not group_model else 3
    unique_groups = grouped[group_by].unique()
    group_chunks = [unique_groups[i:i+max_groups] for i in range(0, len(unique_groups), max_groups)]
    
    lista_cores = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
    categories = sorted(grouped["discipline"].unique()) if group_model else sorted(grouped["model"].unique())
    color_map = {category: lista_cores[i % len(lista_cores)] for i, category in enumerate(categories)}
    
    for group_subset in group_chunks:
        fig, ax = plt.subplots(figsize=(max(12, len(group_subset) * 2), 6))
        width = 0.4
        
        subset_data = grouped[grouped[group_by].isin(group_subset)]
        x_positions = []
        tick_labels = []
        x_index = 0
        spacing = 2  # Aumenta a separação entre grupos distintos
        
        for category in group_subset:
            category_data = subset_data[subset_data[group_by] == category]
            
            if category_data.empty:
                continue
            
            for _, row in category_data.iterrows():
                color = color_map[row["discipline"]] if group_model else color_map[row["model"]]
                ax.bar(x_index, row["Avg_Time"], width, color=color)
                ax.text(x_index, row["Avg_Time"] * 1.03, f"{row['Avg_Time']:.2f}s", ha='center', fontsize=10)
                x_positions.append(x_index)
                tick_labels.append(category)
                x_index += 1
            x_index += spacing  # Adiciona espaço entre os grupos
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels, rotation=45)
        ax.set_ylabel("Tempo Médio (segundos)")
        ax.set_xlabel("Modelos" if group_model else "Disciplinas")
        ax.set_title("Tempo Médio por Disciplina e Modelo" if not group_model else "Tempo Médio por Modelo e Disciplina")
        
        # Criar a legenda corretamente sem duplicação
        unique_labels = sorted(set(color_map.keys()))
        handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in unique_labels]
        ax.legend(handles, unique_labels, title="Disciplinas" if group_model else "Modelos")
        
        plt.show()

def discipline_accuracy_vs_time(df: pd.DataFrame):
    """Gera um gráfico de dispersão mostrando a correlação entre acurácia e tempo médio por disciplina e modelo."""
    df_filtered = df.dropna(subset=["discipline"]).copy()
    df_filtered["time"] = pd.to_numeric(df_filtered["time"], errors="coerce")
    
    grouped = df_filtered.groupby(["model", "discipline"]).agg(
        Avg_Time=("time", "mean"),
        Accuracy=("correct", "mean")
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab10", len(grouped["discipline"].unique()))
    color_map = {discipline: colors(i) for i, discipline in enumerate(sorted(grouped["discipline"].unique()))}
    
    for _, row in grouped.iterrows():
        ax.scatter(row["Avg_Time"], row["Accuracy"], color=color_map[row["discipline"]])
        ax.text(row["Avg_Time"], row["Accuracy"], f"{row['model']}-{row['discipline']}", fontsize=9, ha='right', va='bottom')
    
    ax.set_xlabel("Tempo Médio (segundos)")
    ax.set_ylabel("Acurácia")
    ax.set_title("Correlação entre Acurácia e Tempo Médio por Modelo e Disciplina")
    plt.show()

def multi_model_performance(df: pd.DataFrame, group: str, y_axis: str):
    """Gera um gráfico de barras mostrando a acurácia ou tempo médio por modelo de visão ou modelo de texto."""
    if group not in ["model_vision", "model_text"]:
        raise ValueError("O parâmetro 'group' deve ser 'model_vision' ou 'model_text'")
    if y_axis not in ["time-avg", "accuracy"]:
        raise ValueError("O parâmetro 'y_axis' deve ser 'time-avg' ou 'accuracy'")
    
    df_filtered = df.copy()
    df_filtered["time"] = pd.to_numeric(df_filtered["time"], errors="coerce")
    
    grouped = df_filtered.groupby(group).agg(
        Avg_Time=("time", "mean"),
        Accuracy=("correct", "mean")
    ).reset_index()
    
    y_label = "Tempo Médio (segundos)" if y_axis == "time-avg" else "Acurácia"
    title = "Tempo Médio por Modelo" if y_axis == "time-avg" else "Acurácia por Modelo"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if y_axis == "time-avg":
        bars = ax.bar(grouped[group], grouped["Avg_Time"], color="skyblue")
    else:
        bars = ax.bar(grouped[group], grouped["Accuracy"], color="lightcoral")
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height * 1.03, f"{height:.2f}", ha='center', fontsize=10)
    
    ax.set_xlabel("Modelos")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.show()