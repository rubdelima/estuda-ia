# Estuda-ai

<p align="center">
  <img src="./images/logo.png" alt="estuda-ai" width="30%">
</p>

> **Um ambiente inteligente para a geração e explicação de questões do ENEM utilizando IA**

## **📌 Sobre o Projeto**

O **Estuda-AI** é um projeto desenvolvido por alunos da **Universidade Federal de Pernambuco (UFPE)** durante a disciplina **[IF1006] Tópicos Avançados em Sistemas de Informação 3 - Transformação Digital com IA**. Nosso principal objetivo é criar um ambiente inteligente para auxiliar na geração, explicação e análise de questões do ENEM, utilizando modelos de Inteligência Artificial (IA).

## **Como Executar Localmente**

Primeiramente recomendamos que inicie uma **venv** para executar. Para fazer isso você deve realizar o seguinte comando:

```sh
python3 -m venv .estuda_ai

```

Após criar a venv você pode inicializar ela com os comandos:

**Caso esteja no Windows:**

```sh
./.estuda_ai/Scripts/activate

```
**Caso esteja no Linux ou MacOS:**

```sh
source .estuda_ai/bin/activate
```

Após inicializar a venv instale as dependências com:

```sh
pip install -r requirements.txt
```

E você pode executar o projeto com

```sh
streamlit run app/__main__.py
```

## **🔍 Etapas do Projeto**

1️⃣ **Coleta e Visualização de Dados**

- [Notebook](./01%20-%20Coleta%20e%20Processamento%20de%20Dados.ipynb)
- Construímos um banco de dados com questões do ENEM e suas respectivas resoluções.
- Aplicamos pré-processamento de texto, incluindo tokenização, limpeza e normalização.

2️⃣ **Avaliação de Modelos Locais**

- [Notebook](./02%20-%20Avaliação%20de%20Modelos%20Locais.ipynb)
- Testamos diferentes modelos de IA localmente, utilizando o **Ollama**, analisando desempenho, custo e viabilidade.
- Nosso objetivo foi encontrar os melhores modelos para realizar o fine-tuning posteriormente.

3️⃣ **Avaliação de Modelos via API**

- [Notebook](./02%20-%20Teste%20em%20Modelos%20via%20API.ipynb)
- Testamos diferentes modelos de IA via API, utilizando o **Gemini** e **OpenAI**, analisando desempenho, custo e viabilidade.
- Nosso objetivo foi identificar os modelos mais adequados para o fine-tuning.

4️⃣ **Fine-tuning dos Modelos**

5️⃣ **Testes e Validação**

---

## **🛠️ Tecnologias Utilizadas**

| Tecnologia | Descrição |
|------------|-----------|
| <p align="center"><img src="https://pandas.pydata.org/static/img/pandas.svg" width="120"></p> | **Pandas**: Biblioteca de manipulação e análise de dados, essencial para o processamento das questões do ENEM. |
| <p align="center"><img src="https://numpy.org/images/logo.svg" width="120"></p> | **NumPy**: Biblioteca fundamental para computação científica em Python, utilizada para operações matemáticas e manipulação de arrays. |
| <p align="center"><img src="https://images.seeklogo.com/logo-png/59/2/ollama-logo-png_seeklogo-593420.png" width="120"></p> | **Ollama**: Plataforma de modelos de linguagem que utilizamos para avaliação e fine-tuning de modelos locais. |
| <p align="center"><img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="120"></p> | **Hugging Face**: Biblioteca que fornece ferramentas para processamento de linguagem natural, facilitando o uso e treinamento de modelos de IA. |
| <p align="center"><img src="https://matplotlib.org/_static/images/logo2.svg" width="120"></p> | **Matplotlib**: Biblioteca de plotagem em Python, utilizada para visualização de dados e resultados das análises. |
| <p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Google_Gemini_logo.svg/1280px-Google_Gemini_logo.svg.png" width="120"></p> | **GenAI**: Plataforma utilizada para integração com o modelo Gemini via API, permitindo avaliações de desempenho e viabilidade. |
| <p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/2560px-OpenAI_Logo.svg.png" width="120"></p> | **OpenAI**: Biblioteca Python utilizada para acessar modelos avançados de IA via API, auxiliando na geração e explicação de questões. |
| <p align="center"><img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" width="120"></p> | **Streamlit**: Framework utilizado para o desenvolvimento da interface web interativa do Estuda-AI. |

---

## **👥 Equipe**

- [Erlan Lira Soares Junior](https://github.com/erlanliraa) - elsj
- [Felipe de Barros Moraes](https://github.com/FelipeMoraes03) - fbm3
- [Guilherme Maciel de Melo](https://github.com/GuilhermeMaciel75) - gmm7
- [Rubens Nascimento de Lima](https://github.com/rubdelima) - rnl2

---
