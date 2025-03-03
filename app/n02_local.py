import streamlit as st
import lib.utils as utils
from lib.utils import plots
import pandas as pd
import random
from lib.utils.models_info import models_data
random.seed(42)
from itertools import product
import uuid

def show_table_metrics(models, questions, table):
    table_type = [
        "Desempenho de Acerto x Erros", "Métricas de Tempo por Modelo", "Tempo Total por Modelo",
        "Correlação Tamanho x Acurácia", "Correlação Tempo Médio x Acurácia", "Correlação Tempo x Tamanho",
        "Desempenho por Disciplina (Agrupado por Modelo)", "Desempenho por Disciplina (Agrupado por Disciplina)",
        "Tempo por Disciplina (Agrupado por Modelo)", "Tempo por Disciplina (Agrupado por Disciplina)",
    ]
    
    plot_type = st.selectbox("Seelcione uma forma de visualização: ", table_type, index=0, key='unic' + '-'.join(models))
    
    match(plot_type):
        case "Desempenho de Acerto x Erros":
            return plots.model_performance(table)
        case "Métricas de Tempo por Modelo":
            return plots.time_metrics(table)
        case "Tempo Total por Modelo":
            return plots.time_metrics_total(table)
        case "Correlação Tamanho x Acurácia":
            return plots.correlation(table, x="Size", y="Acc")
        case "Correlação Tempo Médio x Acurácia":
            return plots.correlation(table, x="Tavg", y="Acc")
        case "Correlação Tempo x Tamanho":
            return plots.correlation(table, x="Tavg", y="Size")
        case "Desempenho por Disciplina (Agrupado por Modelo)":
            return plots.discipline_performance(models, questions,True, False)
        case "Desempenho por Disciplina (Agrupado por Disciplina)":
            return plots.discipline_performance(models, questions, False, False)
        case "Tempo por Disciplina (Agrupado por Modelo)":
            return plots.discipline_time_performance(models, questions, True)
        case "Tempo por Disciplina (Agrupado por Disciplina)":
            return plots.discipline_time_performance(models, questions, False)
        case _:
            plots.model_performance(table)

def show_metrics(models,questions):    
    selected_models = st.multiselect("Selecione os modelo desejados: ", models, default=models[0], max_selections=5, key='multi' + '-'.join(models))
    
    table = utils.test_table(questions=questions,models=selected_models)
    
    st.dataframe(utils.format_test_table(table))
    
    plot = show_table_metrics(selected_models, questions, table)
    st.pyplot(plot)


def render(**kwargs):
    questoes = kwargs['questoes']
    
    st.title("🖥️ Fase 1: Teste de Modelos Locais")

    st.markdown("Antes de definirmos qual modelo utilizar em nosso software, decidimos, como primeiro passo, testar modelos open-source disponíveis. O objetivo foi compreender melhor a dinâmica desses modelos e sua aplicabilidade em diferentes cenários. Para isso, criamos um benchmark abrangente, avaliando uma série de modelos com diferentes quantidades de parâmetros, tipos de questões e analisando seu desempenho em diversas disciplinas.")

    st.subheader("1. Modelos")
    
    st.markdown("""Vamos explorar inicialmente modelos que podem ser obtidos utilizando o [Ollama](https://ollama.com/), um repositório de vários modelos de LLM, de forma similar ao Hugging Face, no qual baixamos localmente 20 modelos, variando em alguns a quantidade de parâmetros, os quais podem ser modelos de reasoning, visão computacional, focados em matemática ou apenas texto.""")
    
    models_dataframe = pd.DataFrame(models_data)
    
    st.dataframe(
        models_dataframe.drop(columns=["Imagem", "Descrição"]),
        column_config={
            "Imagem" : st.column_config.ImageColumn(width="medium"),
        },
        use_container_width=True
    )
    
    st.markdown("""
                ### 1.1 Visão Geral dos Modelos
                Abaixo uma leve descrição de cada um dos modelos utilizados""")
    
    for i, model in enumerate(models_dataframe.to_dict(orient="records"), start=1):
        st.markdown(f"#### 1.1.{i} {model['Modelo']}")
        
        cm1, cm2, cm3 = st.columns([2,3,2])
        cm2.image(model['Imagem'])
        
        st.markdown(f"""
            > Parâmetros: {' '.join(map(lambda x : f'`{x}`', model['Parâmetros']))} Tamanho: {' '.join(map(lambda x : f'`{x}GB`', model['Tamanho (Em GB)']))} Algoritmo : `{model['Algoritmo']}`
            
            {model['Descrição']}
            """)
        
    
    st.markdown("""
                ### 1.2 Correlação entre Parâmetros e Tamanho
                Abaixo uma comparação entre o tamanho do modelo (em GB) e a quantidade de parâmetros (em bilhões)""")
    
    st.pyplot(utils.models_info.plot_parameters_x_size())
    
    st.markdown("""
               Observamos que, de maneira geral, **a relação entre a quantidade de parâmetros e o tamanho do modelo segue um padrão bastante linear**. No entanto, há algumas diferenças dependendo do tipo de modelo:  

                - **Modelos de reasoning** tendem a ser **ligeiramente maiores** do que outros modelos com a mesma quantidade de parâmetros.  
                - **Modelos de visão computacional (vision)** também apresentam um **aumento sutil no tamanho**, possivelmente devido às camadas especializadas para processamento de imagens.  

                Essa análise inicial nos permite compreender melhor as exigências de cada modelo em termos de armazenamento e memória, ajudando na escolha da melhor opção para diferentes aplicações locais.
               """)
    
    st.subheader("2. Questões de Texto")
    
    st.markdown("""
                ## 2. Questões de Texto  
                
                O primeiro tipo de questão que vamos analisar são as **questões de texto**, que representam **63,6%** de todas as questões do dataset. Para avaliar o desempenho dos modelos nesse tipo de tarefa, dividimos os testes em **três grupos distintos**:  
                
                1. **Modelos com diferentes versões de tokens** – investigamos o impacto do tamanho do modelo no desempenho, comparando versões de um mesmo modelo com quantidades variadas de tokens.  
                2. **Modelos com suporte a reasoning** – testamos modelos projetados para raciocínio avançado, que, apesar de mais lentos, prometem um processamento mais elaborado e preciso.  
                3. **Demais modelos baseados apenas em texto** – avaliamos modelos que lidam exclusivamente com entrada textual, sem otimizações específicas para reasoning.  
                
                Para conduzir esse experimento, selecionamos **100 questões de forma pseudo-aleatória**, garantindo uma distribuição balanceada com **25 questões de cada disciplina**. Dessa forma, buscamos assegurar que os resultados reflitam de maneira justa a capacidade dos modelos em interpretar e responder diferentes tipos de perguntas textuais.  
            """)
    
    
    # Questões utilizadas no teste
    example_text_questions  = [2011013, 2009066, 2015026, 2014032, 2013088, 2011042, 2010041, 2021090, 2010028, 2023051, 2019054, 2009074, 2009071, 2010033, 2013006, 2014002, 2021048, 2023063, 2009068, 2022062, 2013027, 2019051, 2013063, 2019084, 2023050, 2015058, 2009026, 2012064, 2018134, 2016071, 2012062, 2013079, 2016069, 2011057, 2011051, 2017132, 2011053, 2017104, 2016080, 2023107, 2014089, 2010051, 2019133, 2021118, 2011082, 2017131, 2010081, 2021097, 2015072, 2023099, 2020034, 2016103, 2019044, 2013111, 2022017, 2011103, 2009133, 2021023, 2013099, 2023020, 2014099, 2011109, 2014102, 2011126, 2016115, 2014130, 2017031, 2020045, 2016104, 2012129, 2016108, 2015099, 2013123, 2021030, 2014123, 2021165, 2021142, 2020138, 2009179, 2019153, 2019170, 2011166, 2018161, 2022139, 2013139, 2011162, 2016174, 2015151, 2013166, 2019173, 2021158, 2018169, 2012171, 2021146, 2014168, 2022165, 2022169, 2009175, 2012176, 2009161]

    st.markdown("""
                ### 2.1 Avaliando o Impacto do Tamanho do Modelo  

                Para esse teste, utilizaremos **dois modelos principais**:  

                - **Qwen2.5**, nas versões:  
                  - `14b` (9.0GB)  
                  - `7b` (4.7GB)  
                  - `1.5b` (1.9GB)  

                - **Gemma2**, nas versões:  
                  - `27b` (16GB)  
                  - `9b` (5.4GB)  
                  - `2b` (1.6GB)  

                Escolhemos esses dois modelos porque, entre os selecionados, são os que oferecem **maior diversidade de quantidade de parâmetros**, permitindo uma análise mais abrangente. Além disso, eles apresentam uma variação de tamanho que cobre desde **modelos extremamente compactos** (menos de 2GB, ideais para sistemas embarcados) até **modelos mais convencionais** (`7b` e `9b`), além de **opções mais robustas**, como o `14b` e `27b`.  

                Para esse experimento, utilizaremos o mesmo **conjunto de questões selecionadas de forma pseudo-aleatória**, já descrito anteriormente. Avaliaremos:  

                - **Métricas de desempenho**, incluindo número de acertos e erros, tanto no geral quanto por disciplina.  
                - **Comparação entre os dois modelos**, buscando identificar **tendências de eficiência** em relação à quantidade de parâmetros.  

                Nosso objetivo final é tentar inferir **até que ponto o aumento no número de parâmetros impacta o desempenho** e se há um ponto de equilíbrio entre **qualidade da resposta e eficiência computacional**.  
                """)
    
    st.markdown("""
                #### 2.1.1 - Qwen2.5  

                Nosso primeiro alvo de análise é o **Qwen 2.5**, avaliando seu desempenho em diferentes tamanhos e identificando padrões de comportamento.  

                ##### **Acurácia dos Modelos**  

                Ao analisarmos a **acurácia**, percebemos que **o número de acertos se manteve estável** entre o modelo mais robusto e o modelo intermediário. No entanto, ao compararmos com a versão mais leve, observamos uma **queda significativa**, reduzindo-se para **menos da metade** da acurácia dos demais modelos.  

                ##### **Tempo de Execução**  

                O mesmo padrão foi identificado em relação ao **tempo de execução**. O modelo intermediário apresentou **tempos máximos semelhantes ao modelo maior**, enquanto o modelo mais leve teve uma redução proporcional no tempo de inferência. Além disso, ao analisarmos os tempos médios, identificamos uma **relação linear entre a quantidade de parâmetros e o tempo de execução** – quanto menor o número de parâmetros, menor tende a ser o tempo necessário para processar uma resposta.  

                ##### **Desempenho por Disciplina**  

                Ao compararmos o desempenho por disciplina, observamos que a **diferença entre o modelo intermediário e o modelo maior foi mínima**, exceto por um leve aumento na acurácia em **ciências humanas** no modelo maior. Em **matemática**, ambos os modelos apresentaram **o mesmo desempenho**, sem vantagens claras entre eles.  

                Um padrão recorrente em **todos os tamanhos do Qwen 2.5** foi a **baixa acurácia em questões matemáticas**, o que sugere uma limitação específica do modelo nessa área. Esse comportamento também foi observado em outros modelos, indicando que pode ser uma característica comum entre os LLMs testados.  

                ##### **Visualização dos Resultados**  

                Os gráficos abaixo ilustram esses padrões. Caso esteja acessando via plataforma interativa, você pode **selecionar os modelos e o tipo de gráfico** para comparação personalizada.
                """)
    
    show_metrics(["qwen2.5:14b", "qwen2.5:7b", "qwen2.5:1.5b"], example_text_questions)
    
    st.markdown("""
                #### 2.1.2 - Gemma2  

                Nesta seção, analisamos o desempenho do modelo **Gemma2** em diferentes configurações de tamanho, a fim de identificar padrões de comportamento e avaliar sua eficiência em relação à acurácia e ao tempo de execução.  

                ##### **Acurácia dos Modelos**  

                A análise da **acurácia** revelou que, de maneira similar aos modelos **Qwen2.5**, a quantidade de acertos obtida pelo modelo mais robusto e pelo modelo intermediário permaneceu praticamente inalterada. No entanto, ao comparar com o modelo de menor porte, observou-se uma redução no número de respostas corretas.  

                Diferentemente do que foi constatado no **Qwen2.5**, a perda de desempenho no modelo mais leve do **Gemma2** foi **menos acentuada**. A diferença observada foi de **apenas 10 respostas corretas a menos em relação ao modelo intermediário**, representando uma das melhores relações entre **desempenho e tamanho do modelo** entre os modelos avaliados.  

                ##### **Tempo de Execução**  

                Em relação ao **tempo de inferência**, os resultados obtidos não seguiram a correlação linear observada em outros modelos. O **modelo intermediário apresentou o menor tempo médio e mínimo de execução**, o que diverge da expectativa de um comportamento proporcional ao número de parâmetros. Esse fenômeno sugere que o **Gemma2 pode possuir otimizações internas**, que resultam em um melhor aproveitamento computacional em determinados tamanhos.  

                ##### **Desempenho por Disciplina**  

                A análise do desempenho por disciplina revelou uma **intercalação entre os modelos quanto à sua eficácia em diferentes áreas do conhecimento**:  

                - **Ciências humanas** – Desempenho equivalente entre os modelos maiores, o menor ainda sim conseguiu uma quantidade de acertos aceitáveis, variando em apenas 2 questões.  
                - **Ciências naturais e matemática** – O modelo mais robusto apresentou um desempenho superior, com um aumento de **3 e 2 respostas corretas**, respectivamente.  
                - **Linguagens** – O modelo intermediário superou o modelo mais robusto, obtendo **4 respostas corretas adicionais**.  

                Embora a **acurácia geral tenha se mantido semelhante entre os modelos**, a análise segmentada por disciplina evidencia que **o desempenho dos modelos não é uniforme** e pode variar conforme o domínio da questão analisada.  

                ##### **Desempenho em Matemática**  

                O padrão de **baixo desempenho em matemática** também foi identificado no modelo **Gemma2**, confirmando a tendência observada em outros modelos. Especificamente, os resultados foram:  

                - **Modelo robusto** – 7 acertos  
                - **Modelo intermediário** – 5 acertos  
                - **Modelo leve** – 4 acertos  

                Independentemente da configuração utilizada, o desempenho em matemática manteve-se reduzido, o que sugere uma **limitação inerente ao modelo** no processamento desse tipo específico de questão.  

                ##### **Visualização dos Resultados**  

                As informações detalhadas podem ser observadas nos gráficos apresentados a seguir. Além disso, caso o leitor esteja acessando por meio da plataforma interativa, é possível **selecionar os modelos e o tipo de gráfico desejado** para uma análise comparativa personalizada.  
                """)
    
    show_metrics(["gemma2:2b", "gemma2", "gemma2:27b"], example_text_questions)
    
    st.markdown("""
                #### 2.1.3 - Análise Comparativa dos Modelos  

                Com base nos dados apresentados, observamos que há uma **diferença significativa** de desempenho entre os modelos mais robustos e os modelos mais leves. No entanto, ao compararmos um modelo **intermediário** com um modelo mais robusto, essa diferença torna-se **pouco expressiva**, sugerindo que modelos intermediários podem oferecer um desempenho equivalente na maioria dos cenários. No caso do **Gemma2**, identificamos uma variação entre os acertos por disciplina, com **diferenças mais notáveis entre os modelos**, o que indica que alguns domínios podem ser mais sensíveis à quantidade de parâmetros do modelo.  

                ##### **Impacto Temporal e Considerações Computacionais**  

                Em termos de **tempo de inferência**, o modelo mais robusto apresentou **tempos de execução mais elevados** em todos os testes. Esse comportamento pode ser explicado por dois fatores principais:  

                1. **Consumo de memória** – Modelos maiores tendem a exigir mais memória, sendo preferencialmente executados em **memória de vídeo (VRAM)** para maior eficiência. No entanto, se o modelo ultrapassar a capacidade da GPU disponível, ele será carregado na **memória RAM**, que possui uma **frequência menor**, impactando diretamente no tempo de inferência.  
                2. **Número de combinações avaliadas** – Modelos com maior número de parâmetros possuem **mais possibilidades de ajuste para cada predição**, o que pode levar a um aumento no tempo de resposta.  

                Esses fatores reforçam a necessidade de um equilíbrio entre **qualidade da resposta e viabilidade computacional**, especialmente em aplicações que demandam tempo de resposta reduzido e menor consumo de recursos.  
                ##### **Formulação e Teste da Hipótese Nula**  
                """)
    
    st.subheader("Formulação e Teste da Hipótese Nula")

    st.write("Para avaliar a equivalência entre modelos intermediários e robustos, formulamos a seguinte hipótese estatística:")

    st.latex(r"H_0: \mu_{\text{intermediário}} = \mu_{\text{robusto}}")

    st.write("Ou seja, a **média de acertos do modelo intermediário** ("
             r"$\mu_{\text{intermediário}}$) é estatisticamente equivalente à "
             r"**média de acertos do modelo robusto** ($\mu_{\text{robusto}}$), "
             "indicando que ambos possuem desempenhos similares.")

    st.write("**Hipótese Alternativa ($H_1$):**")

    st.latex(r"H_1: \mu_{\text{intermediário}} \neq \mu_{\text{robusto}}")

    st.write("Essa hipótese alternativa indicaria que existe uma **diferença estatisticamente significativa** "
             "entre os modelos, sugerindo que um deles apresenta um desempenho superior ao outro.")

    st.subheader("Resultados dos Testes Estatísticos")

    st.write("Para validar essa hipótese, aplicamos **dois testes estatísticos**:")
    st.write("- **Teste t de Student**, que avalia se há uma diferença significativa entre os modelos.")
    st.write("- **Teste de Equivalência (TOST)**, que verifica se a diferença de desempenho entre os modelos "
             "está dentro de um intervalo previamente definido como aceitável.")

    st.write("Os resultados obtidos foram os seguintes:")

    st.write("**Teste t de Student ($\\alpha = 0.05$):**")
    st.write("- Estatística t: **-0.1597**")
    st.write("- Valor-p: **0.8732**")
    st.write("- **Conclusão**: Como $p > 0.05$, aceitamos $H_0$, indicando que **não há diferença estatisticamente significativa** entre os modelos intermediário e robusto.")

    st.write("**Teste de Equivalência (TOST) com intervalo de $[-0.02, 0.02]$:**")
    st.write("- Valor-p: **0.3162**")
    st.write("- **Conclusão**: Como $p > 0.05$, **não há evidências suficientes para afirmar equivalência** dentro do intervalo estabelecido.")

    st.subheader("Interpretação dos Resultados")

    st.write("Os resultados obtidos demonstram que **os modelos intermediário e robusto possuem desempenhos estatisticamente similares**, "
             "pois o teste t de Student **não encontrou diferença significativa** entre eles. "
             "No entanto, o teste de equivalência **não conseguiu confirmar que os modelos estão dentro do intervalo de tolerância de ±2%**, "
             "o que significa que **a equivalência estatística não pode ser garantida dentro desse critério específico**.")

    st.write("Dessa forma, podemos concluir que **o aumento no número de parâmetros não implica necessariamente em uma melhoria significativa no desempenho**, "
             "mas também que **não podemos afirmar que um modelo intermediário é equivalente a um robusto dentro da margem de erro estipulada**.")

    st.write("Na prática, a escolha entre um modelo intermediário ou robusto deve considerar não apenas a acurácia estatística, "
             "mas também **fatores como tempo de inferência, consumo de memória e viabilidade computacional**, "
             "uma vez que modelos intermediários podem oferecer benefícios significativos em eficiência sem comprometer substancialmente a qualidade das respostas.")

    
    st.markdown("""
                ### 2.2 - Avaliação de Modelos de Reasoning  

                Nesta seção, analisamos o desempenho dos **modelos de Reasoning**, que, devido à sua estrutura avançada de processamento e raciocínio, são esperados apresentar **melhor desempenho** em tarefas que exigem lógica e cálculos mais complexos. Para esse experimento, utilizamos os seguintes modelos: **Deepscaler**, **Deepseek-r1**, **Mistral-nemo**, **Openthinker** e **Smallthinker** .

                ##### **Tempo de Execução e Uso de Tokens**  

                Observamos uma **diferença significativa no tempo de inferência** desses modelos em comparação com os testados anteriormente. Alguns deles apresentaram a tag `<think>`, que representa uma etapa explícita de raciocínio antes da resposta final. Essa estrutura resultou em **respostas mais longas**, com um maior número de **tokens**, o que impactou diretamente no tempo de processamento.  

                Um caso extremo foi o **Openthinker**, que levou **mais de 1h24min** para processar **100 questões**, resultando em uma média de aproximadamente **50 segundos por questão**. Em contraste, o modelo **Gemma2:27b**, que anteriormente apresentou o maior tempo de execução entre os modelos testados, levou **cerca de 12 minutos** para concluir o mesmo teste.  

                Vale ressaltar que, diferentemente do **Gemma2:27b**, cujo tempo elevado se deve ao alto consumo de **VRAM**, os modelos de reasoning são **mais leves em tamanho**, sugerindo que o tempo prolongado pode estar mais relacionado ao **processamento adicional de raciocínio**. Essa tendência foi observada em **todos os modelos**, exceto pelo **Mistral-nemo**, que conseguiu um tempo de inferência reduzido. 

                Um outro fator importante foi o excesso de **timeout**, com os demais modelos não tivemos problemas, mas o **smallthinker** apresentou uma incostância muito grande, travando a execução, tendo que ser implementado um timeout para impedir que ele interrompesse os teste.

                ##### **Acurácia e Desempenho por Disciplina**  

                Em relação às **métricas de acurácia**, os resultados foram variados:  

                - Apenas o **Deepscaler** apresentou um desempenho **inferior** aos modelos convencionais testados anteriormente.  
                - Os demais modelos tiveram valores **próximos** aos obtidos pelos modelos **não-reasoning**.  
                - Em **matemática**, esperava-se que os modelos reasoning apresentassem **um desempenho superior**, devido à sua **capacidade aprimorada de cálculos**. No entanto, apenas o **Deepseek-r1** obteve **mais de 50% de acertos** nessa disciplina, enquanto nas demais áreas seu desempenho foi inferior, especialmente em **ciências da natureza**.  

                ##### **Respostas Nulas e Impacto no Não-Determinismo**  

                Um fenômeno observado nesses modelos foi a **alta taxa de respostas nulas**, especialmente em **matemática**. Respostas nulas ocorreram quando o modelo **não seguiu corretamente a instrução de resposta**, por exemplo:  

                - Em vez de fornecer a **alternativa correta**, o modelo apresentava o **resultado do cálculo**.  
                - Algumas respostas continham explicações extensas, mas sem indicar diretamente a alternativa correta.  

                Esse comportamento indica que a **camada adicional de raciocínio pode impactar o determinismo do modelo**, tornando suas respostas menos previsíveis e mais propensas a **desvios das instruções originais**.  

                ##### **Considerações Finais**  

                Os modelos de reasoning apresentaram **vantagens e desvantagens** claras. Enquanto alguns conseguiram **bons desempenhos em acurácia**, houve um **custo expressivo em tempo de inferência**. Além disso, a **tendência a respostas mais elaboradas e menos diretas** pode ser um desafio em aplicações que exigem **alta confiabilidade e aderência estrita às instruções**.                  
                """)
    
    show_metrics(["deepscaler", "deepseek-r1", "mistral-nemo", "openthinker", "smallthinker"], example_text_questions)
    
    
    
    
    
    
    
    