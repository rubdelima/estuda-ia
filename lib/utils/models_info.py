import matplotlib.pyplot as plt
import pandas as pd

models = {
    "deepscaler": {
        "parameters": 1.5,
        "size": 3.6,
        "algorithm": "reasoning"
    },
    "mistral-nemo": {
        "parameters": 12.0,
        "size": 7.1,
        "algorithm": "reasoning"
    },
    "mathstral": {
        "parameters": 7.0,
        "size": 4.1,
        "algorithm": "math"
    },
    "qwen2.5:14b": {
        "parameters": 14.0,
        "size": 9.0,
        "algorithm": "text"
    },
    "qwen2.5:7b": {
        "parameters": 7.0,
        "size": 4.7,
        "algorithm": "text"
    },
    "qwen2.5:1.5b": {
        "parameters": 1.5,
        "size": 0.986,
        "algorithm": "text"
    },
    "qwen2-math:7b": {
        "parameters": 7.0,
        "size": 4.4,
        "algorithm": "math"
    },
    "qwen2-math:1.5b": {
        "parameters": 1.5,
        "size": 0.934,
        "algorithm": "math"
    },
    "openthinker": {
        "parameters": 7.0,
        "size": 4.7,
        "algorithm": "reasoning"
    },
    "smallthinker": {
        "parameters": 3.0,
        "size": 3.6,
        "algorithm": "reasoning"
    },
    "phi4": {
        "parameters": 14.0,
        "size": 9.1,
        "algorithm": "text"
    },
    "phi3.5": {
        "parameters": 3.8,
        "size": 2.2,
        "algorithm": "text"
    },
    "gemma2": {
        "parameters": 9.0,
        "size": 5.4,
        "algorithm": "text"
    },
    "gemma2:2b": {
        "parameters": 2.0,
        "size": 1.6,
        "algorithm": "text"
    },
    "gemma2:27b": {
        "parameters": 27.0,
        "size": 16.0,
        "algorithm": "text"
    },
    "llava": {
        "parameters": 7.0,
        "size": 4.7,
        "algorithm": "vision"
    },
    "deepseek-r1": {
        "parameters": 7.0,
        "size": 7.0,
        "algorithm": "reasoning"
    },
    "llama3.2": {
        "parameters": 3.0,
        "size": 2.0,
        "algorithm": "text"
    },
    "llama3.2-vision": {
        "parameters": 11.0,
        "size": 7.9,
        "algorithm": "vision"
    },
    "mistral": {
        "parameters": 7.0,
        "size": 4.1,
        "algorithm": "text"
    },
    "mistral-small": {
        "parameters": 24.0,
        "size": 14.0,
        "algorithm": "text"
    },
    "llava-llama3": {
        "parameters": 8.0,
        "size": 5.5,
        "algorithm": "vision"
    },
    "minicpm-v": {
        "parameters": 8.0,
        "size": 5.5,
        "algorithm": "vision",
    },
    "moondream": {
        "parameters": 1.8,
        "size": 1.7,
        "algorithm": "vision"
    },
    "llava-phi3": {
        "parameters": 3.8,
        "size": 2.9,
        "algorithm": "vision"
    }
}

models_data = {
    "Modelo": [
        "deepseek-r1", "deepscaler", "gemma2", "llava", "llava-llama3", "llava-phi3",
        "llama3.2", "llama3.2-vision", "mathstral", "minicpm-v", "mistral", "mistral-nemo",
        "mistral-small", "moondream", "openthinker", "phi3.5", "phi4", "qwen2-math", "qwen2.5", "smallthinker"
    ],
    "Imagem": [
        "https://images.fastcompany.com/image/upload/f_webp,q_auto,c_fit,w_1024,h_1024/wp-cms-2/2025/01/i-0-91268357-deepseek-logo.jpg",
        "https://avatars.githubusercontent.com/u/174067447?s=200&v=4",
        "https://ollama.com/assets/library/gemma2/58a4be20-b402-4dfa-8f1d-05d820f1204f",
        "https://registry.npmmirror.com/@lobehub/icons-static-png/1.24.0/files/dark/llava-color.png",
        "https://ollama.com/assets/library/llava-llama3/dc3b65cd-62de-45cd-93f9-5c6da62214fa",
        "https://ollama.com/assets/library/llava-llama3/dc3b65cd-62de-45cd-93f9-5c6da62214fa",
        "https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/69129b55-6798-43cd-92b5-0203f5d5a2f3/10.png?t=1730375002",
        "https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/69129b55-6798-43cd-92b5-0203f5d5a2f3/10.png?t=1730375002",
        "https://ollama.com/assets/library/mathstral/d21307b1-fe6d-4ca6-ab07-f2482a75cdca",
        "https://ollama.com/assets/library/minicpm-v/9252c73d-2c9c-434c-8a34-21f4d5cdd25e",
        "https://github.com/jmorganca/ollama/assets/3325447/d6be0694-eb35-417b-8f08-47d3b6c2a171",
        "https://github.com/jmorganca/ollama/assets/3325447/d6be0694-eb35-417b-8f08-47d3b6c2a171",
        "https://github.com/jmorganca/ollama/assets/3325447/d6be0694-eb35-417b-8f08-47d3b6c2a171",
        "https://replicate.delivery/pbxt/KZKNhDQHqycw8Op7w056J8YTX5Bnb7xVcLiyB4le7oUgT2cY/moondream2.png",
        "https://ollama.com/assets/library/openthinker/0ef3a0d3-aae1-4855-b56b-50875e9683e8",
        "https://media.licdn.com/dms/image/v2/D4D12AQE-07X1ijhiFg/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1716483690066?e=2147483647&v=beta&t=9BUj5e8xKzVvHQcK2LHsCwwl7F5EPLxM_DU8bxff-kE",
        "https://www.aisharenet.com/wp-content/uploads/2025/01/bbde9e53c7505db.jpg",
        "https://ollama.com/assets/library/qwen2-math/c0159f63-bf4d-4b0d-9b92-ea44c48d34c1",
        "https://ollama.com/assets/library/qwen2.5/4b4f719f-c327-489e-8dc1-89a455c21e89",
        "https://ollama.com/assets/library/smallthinker/1d25cb29-e27d-492c-be53-ce79b20def5b"
    ],
    "Parâmetros": [
        [7.0], [1.5], [2.0, 9.0, 27], [7.0], [8.0], [3.8], [3.0], [11.0], [7.0], [8.0],
        [7.0], [12.0], [24.0], [1.8], [7.0], [3.8], [14.0], [7.0, 1.5], [14.0, 7.0, 1.5], [3.0]
    ],
    "Tamanho (Em GB)": [
        [7.0], [3.6], [1.6, 5.4, 16.0], [4.7], [5.5], [2.9], [2.0], [7.9], [4.1], [5.5],
        [4.1], [7.1], [14.0], [1.7], [4.7], [2.2], [9.1], [4.4, 0.934], [9.0, 4.7, 0.986], [3.6]
    ],
    "Algoritmo": [
        "reasoning", "reasoning", "text", "vision", "vision", "vision", "text", "vision",
        "math", "vision", "text", "reasoning", "text", "vision", "reasoning", "reasoning",
        "text", "math", "text", "reasoning"
    ],
    "Descrição": [
        "Primeira geração de modelos de raciocínio do DeepSeek com desempenho comparável ao OpenAI-o1, incluindo seis modelos densos extraídos do DeepSeek-R1 com base no Llama e no Qwen.",
        "Uma versão de fine-tunning do Deepseek-R1-Distilled-Qwen-1.5B que supera o desempenho do o1-preview da OpenAI com apenas 1.5B parâmetros em avaliações matemáticas populares.",
        "O Google Gemma 2 é um modelo eficiente e de alto desempenho, projetado pela Google, com uma arquitetura totalmente nova, projetada para desempenho e eficiência líderes de mercado.",
        "🌋 LLaVA é um novo modelo multimodal grande treinado de ponta a ponta que combina um codificador de visão e Vicuna para compreensão visual e de linguagem de uso geral. Atualizado para a versão 1.6.",
        "Um modelo LLaVA aprimorado do Llama 3 Instruct com melhores pontuações em vários benchmarks.",
        "llava-phi3 é um modelo LLaVA aprimorado a partir do Phi 3 Mini 4k, com fortes benchmarks de desempenho no mesmo nível do modelo LLaVA original",
        "A coleção Meta Llama 3.2 de modelos multilíngues de grandes linguagens (LLMs) é uma coleção de generativos pré-treinados e ajustados por instruções. Os modelos somente texto ajustados por instruções Llama 3.2 são otimizados para casos de uso de diálogo multilíngue, incluindo tarefas de recuperação e sumarização de agentes.",
        "Llama 3.2 Vision é uma coleção de modelos generativos de raciocínio de imagem ajustados por instruções em tamanhos 11B e 90B. Eles são otimizados para reconhecimento visual, raciocínio de imagem, legendas e respostas a perguntas gerais sobre uma imagem.",
        "Mistral AI está contribuindo com o Mathstral para a comunidade científica para reforçar os esforços em problemas matemáticos avançados que exigem raciocínio lógico complexo e multietapas. O lançamento do Mathstral é parte de seu esforço mais amplo para dar suporte a projetos acadêmicos — foi produzido no contexto da colaboração da Mistral AI com o Projeto Numina.",
        "O MiniCPM-V 2.6 é o modelo mais recente e capaz da série MiniCPM-V. Ele apresenta novos recursos para compreensão de várias imagens e vídeos. Os recursos notáveis do MiniCPM-V 2.6 incluem: Desempenho líder, compreensão de várias imagens e aprendizado em contexto, forte capacidade de OCR e eficiência superior, além de seu tamanho amigável.",
        "Mistral é um modelo de parâmetro 7B, distribuído com a licença Apache. Ele está disponível tanto em instruct (seguimento de instruções) quanto em text completion.",
        "Mistral NeMo é um modelo 12B construído em colaboração com a NVIDIA. Mistral NeMo oferece uma grande janela de contexto de até 128k tokens. Seu raciocínio, conhecimento de mundo e precisão de codificação são de última geração em sua categoria de tamanho. Como depende de arquitetura padrão, Mistral NeMo é fácil de usar e um substituto imediato em qualquer sistema que use Mistral 7B.",
        "O Mistral Small 3 estabelece um novo padrão na categoria de modelos de linguagem grandes “pequenos” abaixo de 70B, ostentando parâmetros de 24B e alcançando recursos de última geração comparáveis a modelos maiores.",
        "🌔 moondream2 é um pequeno modelo de linguagem de visão projetado para rodar eficientemente em dispositivos menores.",
        "Uma família de modelos de raciocínio totalmente de código aberto construída usando um conjunto de dados a partir de fine-tunning do Qwen2.5, superando o DeepSeek-R1 em alguns benchmarks",
        "Phi-3.5-mini é um modelo aberto leve e de última geração, criado com base em conjuntos de dados usados ​​para Phi-3 — dados sintéticos e sites filtrados disponíveis publicamente, com foco em dados de altíssima qualidade e densos em raciocínio.",
        "Phi-4 é um modelo aberto de última geração, com 14B parâmetros, construído sobre uma mistura de conjuntos de dados sintéticos, dados de sites de domínio público filtrados e livros acadêmicos adquiridos e conjuntos de dados de perguntas e respostas.",
        "Qwen2 Math é uma série de modelos de linguagem matemática especializados desenvolvidos com base nos Qwen2 LLMs, que superam significativamente as capacidades matemáticas de modelos de código aberto e até mesmo modelos de código fechado (por exemplo, GPT4o).",
        "Os modelos Qwen2.5 são pré-treinados no mais recente conjunto de dados de larga escala do Alibaba, abrangendo até 18 trilhões de tokens. O modelo suporta até 128K tokens e tem suporte multilíngue.",
        "Um novo modelo de raciocínio pequeno, ajustado a partir do modelo Qwen 2.5 3B Instruct.",
    ],
}


def plot_parameters_x_size():
    # Dicionário de cores para cada algoritmo
    colors = {
        "reasoning": "red",
        "math": "blue",
        "text": "green",
        "vision": "orange"
    }

    # Preparar os dados para o plot a partir do dicionário global 'models'
    parameters = [models[model]["parameters"] for model in models]
    sizes = [models[model]["size"] for model in models]
    algorithms = [models[model]["algorithm"] for model in models]

    # Cria a figura e o eixo
    fig, ax = plt.subplots(figsize=(12, 8))

    # Para cada algoritmo distinto, plota os pontos correspondentes
    for alg in set(algorithms):
        indices = [i for i, a in enumerate(algorithms) if a == alg]
        ax.scatter(
            [parameters[i] for i in indices],
            [sizes[i] for i in indices],
            label=alg,
            color=colors[alg]
        )

    ax.set_xlabel('Parâmetros (Bilhões)')
    ax.set_ylabel('Tamanho (GB)')
    ax.set_title('Parâmetros vs. Tamanho dos Modelos')
    ax.legend()
    plt.tight_layout()

    return fig
