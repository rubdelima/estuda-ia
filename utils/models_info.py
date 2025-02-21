import matplotlib.pyplot as plt

models = {
    "deepscaler" : {
        "parameters" : 1.5,
        "size" : 3.6,
        "algorithm" : "reasoning"
    },
    "mistral-nemo" : {
        "parameters" : 12.0,
        "size" : 7.1,
        "algorithm" : "reasoning"
    },
    "mathstral" : {
        "parameters" : 7.0,
        "size" : 4.1,
        "algorithm" : "math"
    },
    "qwen2.5:14b" : {
        "parameters" : 14.0,
        "size" : 9.0,
        "algorithm" : "text"
    },
    "qwen2.5:7b" : {
        "parameters" : 7.0,
        "size" : 4.7,
        "algorithm" : "text"
    },
    "qwen2.5:1.5b" : {
        "parameters" : 1.5,
        "size" : 0.986,
        "algorithm" : "text"
    },
    "qwen2-math:7b" : {
        "parameters" : 7.0,
        "size" : 4.4,
        "algorithm" : "math"
    },
    "qwen2-math:1.5b" : {
        "parameters" : 1.5,
        "size" : 0.934,
        "algorithm" : "math"
    },
    "openthinker":{
        "parameters" : 7.0,
        "size" : 4.7,
        "algorithm" : "reasoning"
    },
    "smallthinker": {
        "parameters" : 3.0,
        "size" : 3.6,
        "algorithm" : "reasoning"
    },
    "phi4" : {
        "parameters" : 14.0,
        "size" : 9.1,
        "algorithm" : "text"
    },
    "phi3.5" : {
        "parameters" : 3.8,
        "size" : 2.2,
        "algorithm" : "text"
    },
    "gemma2" : {
        "parameters" : 9.0,
        "size" : 5.4,
        "algorithm" : "text"
    },
    "llava" : {
        "parameters" : 7.0,
        "size" : 4.7,
        "algorithm" : "vision"
    },
    "deepseek-r1" : {
        "parameters" : 7.0,
        "size" : 7.0,
        "algorithm" : "reasoning"
    },
    "llama3.2" : {
        "parameters" : 3.0,
        "size" : 2.0,
        "algorithm" : "text"
    },
    "llama3.2-vision" :{
        "parameters" : 11.0,
        "size" : 7.9,
        "algorithm" : "vision"
    },
    "mistral" :{
        "parameters" : 7.0,
        "size" : 4.1,
        "algorithm" : "text"
    },
    "mistral-small" : {
        "parameters" : 24.0,
        "size" : 14.0,
        "algorithm" : "text"
    },
    "llava-llama3" : {
        "parameters" : 8.0,
        "size" : 5.5,
        "algorithm" : "vision"
    },
    "minicpm-v" :{
        "parameters" : 8.0,
        "size" : 5.5,
        "algorithm" : "vision",
    },
    "moondream" : {
        "parameters" : 1.8,
        "size" : 1.7,
        "algorithm" : "vision"
    },
    "llava-phi3" : {
        "parameters" : 3.8,
        "size" : 2.9,
        "algorithm" : "vision"
    }
}

def plot_parameters_x_size():
    # Cores para cada algoritmo
    colors = {
        "reasoning": "red",
        "math": "blue",
        "text": "green",
        "vision": "orange"
    }

    # Preparar os dados para o plot
    parameters = [models[model]["parameters"] for model in models]
    sizes = [models[model]["size"] for model in models]
    algorithms = [models[model]["algorithm"] for model in models]
    color_labels = [colors[alg] for alg in algorithms]

    # Plot
    plt.figure(figsize=(12, 8))
    for alg in set(algorithms):
        indices = [i for i, a in enumerate(algorithms) if a == alg]
        plt.scatter(
            [parameters[i] for i in indices],
            [sizes[i] for i in indices],
            label=alg,
            color=colors[alg]
        )
    plt.xlabel('Parâmetros (Bilhões)')
    plt.ylabel('Tamanho (GB)')
    plt.title('Parâmetros vs. Tamanho dos Modelos')
    plt.legend()
    plt.show()