import streamlit as st
import lib.utils as utils
from lib.utils import plots
import pandas as pd
import random
from lib.utils.models_info import models_data
random.seed(42)


def show_table_metrics(models, questions, table):
    table_type = [
        "Desempenho de Acerto x Erros", "Métricas de Tempo por Modelo", "Tempo Total por Modelo",
        "Correlação Tamanho x Acurácia", "Correlação Tempo Médio x Acurácia", "Correlação Tempo x Tamanho",
        "Desempenho por Disciplina (Agrupado por Modelo)", "Desempenho por Disciplina (Agrupado por Disciplina)",
        "Tempo por Disciplina (Agrupado por Modelo)", "Tempo por Disciplina (Agrupado por Disciplina)",
        "Diagrama de Venn"
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
        case "Diagrama de Venn":
            if len(questions) < 2:
                st.info("Você precisa selecionar pelo menos duas disciplinas para visualizar o diagrama de Venn.")
                return None
            if len(questions) > 3:
                st.info("Você pode escolher até 3 disciplinas para visualizar o diagrama de Venn.")
                return None
            return plots.venn_diagram(models, *questions)
        case _:
            plots.model_performance(table)

def show_metrics(models,questions):    
    selected_models = st.multiselect("Selecione os modelo desejados: ", models, default=models[0], max_selections=5, key='multi' + '-'.join(models))
    
    table = utils.test_table(questions=questions,models=selected_models)
    
    st.dataframe(utils.format_test_table(table))
    
    plot = show_table_metrics(selected_models, questions, table)
    st.pyplot(plot)


def show_discipline_metrics(table, discipline):
    
    plot_type = st.selectbox("Seelcione uma forma de visualização: ", ["Questões Acertadas", "Tempo Médio"], index=0, key=discipline)
    
    st.pyplot(plots.discipline_models(table, discipline, 'acc' if plot_type == "Questões Acertadas" else 'time'))


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
        st.markdown(f"#### 1.1.{i} {model['Modelo']}", )
        
        cm1, cm2 = st.columns([2,3], vertical_alignment='center')
        cm1.image(model['Imagem'])
        cm2.markdown(f"""
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

                Para esse teste, utilizaremos **dois modelos principais**, o **Qwen2.5**, nas versões: `14b` (9.0GB), `7b` (4.7GB) e `1.5b` (1.9GB), como também o **Gemma2**, nas versões `27b` (16GB), `9b` (5.4GB) e`2b` (1.6GB). Escolhemos esses dois modelos porque, entre os selecionados, são os que oferecem **maior diversidade de quantidade de parâmetros**, permitindo uma análise mais abrangente. Além disso, eles apresentam uma variação de tamanho que cobre desde **modelos extremamente compactos** (menos de 2GB, ideais para sistemas embarcados) até **modelos mais convencionais** (`7b` e `9b`), além de **opções mais robustas**, como o `14b` e `27b`.  

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
    
    st.markdown("""
                ### 2.3 - Avaliação de Modelos Focados em Matemática  

                Nesta seção, analisamos o desempenho de modelos especializados em matemática, avaliando se sua otimização para essa área impacta significativamente a acurácia em diferentes disciplinas. Os modelos testados foram o **Mathstral**, uma variação do **Mistral 7B** com aprimoramento em ciências da natureza e matemática, e o **Qwen2-Math**, uma versão especializada do **Qwen2**, testada nas configurações de **1.5B** e **7B** parâmetros.  

                Ao comparar esses modelos com os testados anteriormente, observamos que sua acurácia geral foi inferior, sugerindo que, apesar do foco em matemática, seu desempenho em outras disciplinas foi prejudicado. Em especial, notamos um desempenho bastante reduzido em linguagens, onde o maior número de acertos registrado foi de apenas **11**, um resultado significativamente inferior ao dos modelos intermediários analisados previamente.  

                Além da acurácia, analisamos também o tempo de inferência. Os resultados demonstram que as duas versões do **Qwen2-Math** apresentaram uma diferença mínima em termos de acurácia, no entanto, a versão **1.5B teve um tempo médio de inferência muito superior** ao da versão **7B**. Já o **Mathstral**, que possui um tamanho similar ao **Qwen2-Math:7B**, levou **mais do que o dobro do tempo** para concluir o teste, o que indica que diferenças arquiteturais podem ter impactado significativamente sua eficiência computacional.  

                Embora a expectativa fosse de que esses modelos apresentassem um desempenho significativamente superior em matemática, os resultados mostraram que essa melhora não foi expressiva. O modelo que obteve o melhor desempenho na disciplina foi o **Mathstral7b**, porém, de forma surpreendente, o **Qwen2-Math:1.5B**, que, mesmo possuindo **uma quantidade muito inferior de parâmetros**, superou o desempenho de sua variação de parâmetros maior, e ainda conseguiu ser mais rápido como cidato anteriormente.  

                Diante desses resultados, é possível concluir que modelos especializados em matemática **não necessariamente garantem um desempenho superior na disciplina** e, em contrapartida, apresentam **quedas expressivas em outras áreas**, tornando-os menos versáteis. Além disso, a relação entre **tamanho do modelo, tempo de inferência e acurácia** nem sempre segue um padrão linear, como evidenciado pelo desempenho do **Qwen2-Math:1.5B**, que, apesar de menor, apresentou os melhores resultados em sua categoria.
                """)
    
    show_metrics(["qwen2-math:1.5b", "qwen2-math:7b", "mathstral"], example_text_questions)
    
    st.markdown("""
                ### 2.4 - Avaliação dos Demais Modelos  

                Por fim, realizamos testes com os modelos restantes focados em **processamento de texto**, além de um modelo com **suporte à visão**, com o objetivo de avaliar o desempenho de um modelo multimodal. Esses resultados servirão como base para comparações futuras, especialmente quando analisarmos o desempenho dos modelos em questões que envolvem imagens.  

                Os modelos que obtiveram os melhores resultados em **acurácia geral** foram o **Phi-4** e o **Mistral-Small**, ambos apresentando **mais de 70% de precisão**. No entanto, observamos que o **Mistral-Small teve um tempo de execução significativamente superior aos demais**, levando **aproximadamente 44 minutos** para concluir o teste. Além disso, os modelos desenvolvidos pela **Microsoft** também apresentaram **tempos de inferência mais elevados**, embora **não tão acentuados quanto o Mistral-Small**.  

                Um fenômeno incomum foi observado no modelo **Phi-3.5**. Durante as primeiras questões, o modelo demonstrou **bom desempenho**, respondendo de maneira coerente. No entanto, conforme o teste progrediu, o modelo começou a apresentar **alucinações**, gerando respostas totalmente distintas das solicitadas. Esse comportamento impactou diretamente o número de **respostas nulas**, resultando em uma taxa alarmante de **83% de questões inválidas**.  

                Ao analisarmos o gráfico de **correlação entre tempo médio e acurácia**, constatamos que o modelo que obteve **o melhor equilíbrio entre desempenho e tempo de execução** foi o **Phi-4**, pois alcançou **excelentes resultados sem comprometer significativamente o tempo de processamento**. Por outro lado, o desempenho do **Mistral-Small** pode ter sido influenciado pela **limitação de hardware**. Como o modelo possui um tamanho de **14GB**, mas foi executado em uma **placa de vídeo com 12GB de VRAM**, parte do processamento foi deslocado para a **memória RAM**, resultando em um maior tempo de carregamento e execução. Esse fator pode ter impactado diretamente sua velocidade e eficiência durante o teste.  

                Ao analisarmos o desempenho por disciplina, os modelos **Phi-4** e **Mistral-Small** surpreenderam ao apresentar **um desempenho superior em matemática** quando comparados aos modelos projetados especificamente para essa área. O **Phi-4**, em particular, atingiu **92% de precisão** em matemática, mantendo **um desempenho consistente em todas as demais disciplinas**, o que indica sua **elevada capacidade geral** em relação aos outros modelos testados.  

                Outro dado relevante foi a comparação entre o modelo **Mistral** e sua variação especializada em matemática (**Mathstral**). O **Mathstral**, que passou por um processo de fine-tuning voltado para cálculos e raciocínio lógico, demonstrou um **desempenho significativamente superior** ao modelo base. Enquanto o **Mathstral** obteve **12 acertos em matemática**, o **Mistral padrão acertou apenas 3**, evidenciando uma melhora de **400%**. Esse resultado confirma que o fine-tuning foi **bem-sucedido**, tornando o modelo especializado **mais eficiente na resolução de problemas matemáticos**. Em outras disciplinas, no entanto, os dois modelos apresentaram desempenhos variados, com o modelo base tendo um acero a mais em ciências da natureza e 5 a mais em linguagens, enquando o modelo com fine-tunnig mostrou um desempenho superior, além do já mencionado, em ciências humanas. 

                Um dos modelos que se destacou em termos de **desempenho global** foi o **Phi-4**. Seu desempenho nas diferentes disciplinas foi **consistente e expressivo**. O modelo obteve **92% de acertos em ciências humanas**, **76% em linguagem**, **84% em ciências da natureza** e **um impressionante 92% em matemática**, consolidando-se como um dos melhores modelos em termos de **acurácia geral**. Esse resultado indica que, mesmo com seu comportamento instável em longas execuções, o **Phi-3.5 demonstrou um nível de compreensão bastante elevado**, sendo capaz de lidar com questões de diferentes áreas com alto índice de acertos.  

                De maneira geral, os modelos analisados nesta seção demonstraram **desempenhos distintos dependendo do critério avaliado**. Enquanto **Phi-4** se destacou pelo equilíbrio entre **tempo de inferência e acurácia**, **Mistral-Small** apresentou resultados robustos, porém com um custo computacional elevado. O modelo **Phi-3.5**, por sua vez, revelou um desempenho impressionante em todas as disciplinas, especialmente em matemática, mas sofreu com um número elevado de respostas inválidas ao longo do teste. Essas diferenças evidenciam que a escolha do modelo ideal deve levar em consideração **não apenas a acurácia bruta, mas também o tempo de inferência, estabilidade das respostas e adequação às tarefas específicas**.                 
                """)
    
    show_metrics(["phi4", "phi3.5", "llava", "llama3.2", "mistral", "mistral-small"], example_text_questions)
    
    st.markdown("""
                ### 2.5 - Análise de Desempenho por Disciplina  
                
                Com a finalização dos testes para cada categoria de modelo, passamos agora a uma análise mais detalhada do **desempenho por disciplina**. O objetivo é compreender como os diferentes modelos se comportam em cada área do conhecimento, identificando padrões, pontos fortes e eventuais limitações em suas respostas.  
                """)
    
    all_text_models = ["mistral", "phi4", "phi3.5", "llava", "llama3.2",  "mistral-small", "qwen2-math:1.5b", "qwen2-math:7b", "mathstral", "deepscaler", "deepseek-r1", "mistral-nemo", "openthinker", "smallthinker", "gemma2:2b", "gemma2", "gemma2:27b", "qwen2.5:14b", "qwen2.5:7b", "qwen2.5:1.5b"]
    text_table = utils.tabela_geral(example_text_questions, all_text_models)
    table_disciplinas = utils.analisar_tabela(text_table, 'discipline')
    st.dataframe(table_disciplinas)
    
    
    st.markdown("#### 2.5.1 - Matemática")
    
    show_discipline_metrics(text_table, 'matematica')
    
    st.markdown("""
                Em relação ao tempo de execução, as questões de **matemática** apresentaram um tempo médio significativamente superior ao das demais disciplinas, atingindo em média 20,48 segundos por questão, enquanto nas outras áreas do conhecimento nenhuma ultrapassou 8 segundos em média. Essa diferença evidencia que os modelos demandam um tempo consideravelmente maior para processar e responder questões matemáticas, possivelmente devido à necessidade de operações aritméticas ou dificuldades na interpretação das expressões.
                
                Além disso, o tempo máximo registrado para uma única questão de matemática foi de **542 segundos**, um valor muito superior ao tempo máximo observado em outras disciplinas, cujo segundo maior tempo registrado foi de **173,49 segundos**. Essa discrepância sugere que certas questões matemáticas exigem **muito mais processamento**, podendo ativar mecanismos internos dos modelos, como **passos adicionais de raciocínio, tentativa de cálculos detalhados ou maior esforço na busca por padrões matemáticos**. Esse comportamento não foi observado com a mesma intensidade em disciplinas como **linguagens e ciências humanas**, onde as respostas parecem ser geradas com maior rapidez e consistência.  
                
                Esses resultados reforçam a hipótese de que questões matemáticas **impõem uma carga computacional maior** aos modelos, tornando sua resolução **mais lenta e sujeita a variações** quando comparada a outras áreas do conhecimento.  
                """)
    
    st.markdown("#### 2.5.2 - Ciências Humanas")
    show_discipline_metrics(text_table, 'ciencias-humanas')
    st.markdown("""
                A disciplina de **Ciências Humanas** apresentou um desempenho **superior à maioria das demais áreas**, com uma **acurácia média de 67%** entre os modelos testados. Além disso, o **tempo médio de inferência** foi relativamente **baixo**, sendo **muito próximo ao da disciplina de Linguagens**, com uma diferença de apenas **0,0139 segundos**.  

                O bom desempenho nessa disciplina pode estar relacionado ao **tipo de questão abordada**, que, em muitos casos, exige uma resposta mais **direta e factual**, reduzindo a necessidade de **interpretação subjetiva**. Isso pode ter facilitado a geração de respostas corretas pelos modelos, diferentemente do que ocorre em disciplinas que exigem um maior nível de inferência contextual.  

                Embora a média geral tenha sido positiva, alguns modelos se destacaram de forma significativa. **Oito modelos** conseguiram acertar **20 ou mais questões**, um resultado expressivo quando comparado a outras disciplinas. Esse dado reforça que **os modelos de linguagem estão relativamente bem ajustados para esse tipo de conteúdo**, conseguindo lidar com conceitos históricos, geográficos e sociais de maneira eficaz.  
                """)
    
    st.markdown("#### 2.5.3 - Linguagens")
    show_discipline_metrics(text_table, 'linguagens')
    
    st.markdown("""
                O desempenho dos modelos na disciplina de **Linguagens** foi inferior ao de Ciências Humanas, registrando uma **acurácia média de 55,8%**. Apesar disso, foi uma das **melhores métricas entre todas as disciplinas avaliadas**, superando, por exemplo, Matemática e Ciências da Natureza.  
                
                Acreditamos que essa diferença de desempenho em relação a Ciências Humanas pode estar associada à **necessidade de interpretação mais complexa** das questões. Enquanto muitas perguntas em Ciências Humanas possuem **respostas factuais mais objetivas**, as questões de Linguagens frequentemente exigem **análise contextual, compreensão de nuances textuais e inferências**, aspectos que podem representar um desafio adicional para os modelos testados.  
                
                Mesmo com uma média relativamente inferior, **alguns modelos demonstraram desempenho satisfatório**. Apenas **um modelo** conseguiu acertar **mais de 20 questões**, mas **12 modelos** atingiram **15 ou mais acertos**, o que representa um resultado bastante positivo. Esse desempenho sugere que, embora a disciplina de Linguagens represente um desafio maior em termos de interpretação, **boa parte dos modelos ainda conseguiu alcançar uma performance sólida**, demonstrando certo nível de capacidade de compreensão textual. 
                """)
    
    
    st.markdown("#### 2.5.4 - Ciências da Natureza")
    show_discipline_metrics(text_table, 'ciencias-natureza')
    st.markdown("""
                A disciplina de Ciências da Natureza apresentou um desempenho próximo ao observado em Linguagens, tanto em relação à acurácia quanto ao tempo de inferência. No entanto, ao analisarmos a variação de tempo entre os modelos testados, nota-se que a escala de crescimento do tempo de inferência foi exponencialmente inferior à observada em Matemática, sugerindo que a complexidade das questões não gerou um impacto tão significativo no processamento dos modelos. Em termos médios, os modelos levaram aproximadamente dez segundos a mais para responder às questões dessa disciplina quando comparados a Linguagens, enquanto a acurácia média se mostrou apenas um ponto percentual superior.  

                Esse comportamento já era, de certa forma, esperado, uma vez que muitas questões de Ciências da Natureza exigem conhecimento prévio de conceitos específicos, assim como ocorre em Ciências Humanas. Entretanto, diferentemente dessa última, a disciplina incorpora um número significativo de questões que demandam cálculos, aproximando-se, em parte, das exigências matemáticas. Em Física, por exemplo, modelos precisaram lidar com questões relacionadas às leis de Newton e fenômenos ondulatórios, enquanto em Química foram exigidos cálculos estequiométricos e proporções entre reagentes. Até mesmo em Biologia, verificou-se a necessidade de operações numéricas, como cálculos de porcentagens em genética e estatísticas populacionais.  

                Dessa forma, os resultados indicam que Ciências da Natureza pode ser caracterizada como uma disciplina híbrida, combinando elementos de interpretação conceitual e resolução de problemas quantitativos. No entanto, ao considerar o tempo de inferência e a taxa de acertos, observa-se que o comportamento dos modelos nessa disciplina se assemelha mais ao das demais áreas do conhecimento do que ao da Matemática, na qual a escalada do tempo de execução se mostrou muito mais acentuada.  
                """)
    
    
    st.markdown("""
                ### 2.6 - Resultados por Questões  

                Nesta seção, analisamos o desempenho dos modelos sob a ótica das questões individuais, observando a distribuição de acertos e o tempo de inferência médio por questão.  

                Inicialmente, ao considerarmos a distribuição de acertos, observamos que, em média, uma questão foi respondida corretamente por **8,48 modelos**, enquanto a mediana situa-se entre **9 e 15 modelos**. A questão com maior número de acertos foi corretamente respondida por mais de **15 modelos diferentes**, evidenciando que algumas perguntas são resolvidas com alto grau de consistência entre os modelos. Um aspecto interessante é que **não houve nenhuma questão que não tenha sido corretamente respondida por pelo menos um modelo**, conforme ilustrado na distribuição do histograma abaixo. Como pode ser observado, as duas questões com **menor número de acertos** foram resolvidas corretamente por apenas **dois modelos**.  

                A análise desse histograma revela um comportamento que **se aproxima de uma distribuição normal**, com maior concentração de frequência na região central da curva e menor nas extremidades. No entanto, nota-se uma assimetria pontual, especialmente na **categoria de acertos igual a 4**, que apresentou uma frequência superior à categoria seguinte, o que indica uma leve irregularidade na distribuição dos acertos.  
                """)
    
    table_question = utils.analisar_tabela(text_table, 'question')
    table_question = table_question[table_question['Total'] > 10]
    
    st.pyplot(plots.histogram(table_question, 'OK', 100))
    
    st.markdown("Por outro lado, ao analisarmos a distribuição do tempo médio de inferência por questão, observamos um padrão **exponencial**, no qual os tempos mais baixos representam a maior parte da distribuição. A mediana situa-se em **5,4 segundos**, com a média em **9,3 segundos** e o terceiro quartil em **10,66 segundos**, enquanto o tempo máximo registrado foi de **45 segundos**. Esse comportamento sugere que, na maioria dos casos, os modelos conseguem responder rapidamente às questões, exceto em algumas exceções nas quais o tempo de inferência se eleva consideravelmente. Como analisado anteriormente, essas exceções ocorrem com maior frequência em **questões matemáticas**, que exigem processamento adicional, podendo resultar em tempos significativamente mais elevados.")
    
    st.pyplot(plots.histogram(table_question, 'Tavg', 4))
    
    st.markdown("Para ilustrar melhor esses resultados, a seguir apresentamos exemplos das **questões mais fáceis e mais difíceis**, evidenciando os padrões de acerto e tempo de resposta em diferentes tipos de problemas.")
    
    st.markdown("""
                As duas questões que tiveram apenas 1 acerto foram uma de Linguagens e outra de Ciências da Natureza, respectivamente dos anos de 2013 e 2018.
                
                ##### Qustão Difícil: Questão 2013123 - Linguagens
                
                **Para Carr, internet atua no comércio da distração**

                _Autor de “A Geração Superficial” analisa a influência da tecnologia na mente_

                O jornalista americano Nicholas Carr acredita que a internet não estimula a inteligência de ninguém. O autor explica descobertas científicas sobre o funcionamento do cérebro humano e teoriza sobre a influência da internet em nossa forma de pensar.  
                Para ele, a rede torna o raciocínio de quem navega mais raso, além de fragmentar a atenção de seus usuários.  
                Mais: Carr afirma que há empresas obtendo lucro com a recente fragilidade de nossa atenção. “Quanto mais tempo passamos _on-line_ e quanto mais rápido passamos de uma informação para a outra, mais dinheiro as empresas de internet fazem”, avalia.  
                “Essas empresas estão no comércio da distração e são _experts_ em nos manter cada vez mais famintos por informação fragmentada em partes pequenas. É claro que elas têm interesse em nos estimular e tirar vantagem da nossa compulsão por tecnologia.”

                ROXO, E. **Folha de S. Paulo**, 18 fev. 2012 (adaptado).

                (A) : Mantém os usuários cada vez menos preocupados com a qualidade da informação.

                (B) : Torna o raciocínio de quem navega mais raso, além de fragmentar a atenção de seus usuários.

                (C) : Desestimula a inteligência, de acordo com descobertas científicas sobre o cérebro.

                (D) : Influencia nossa forma de pensar com a superficialidade dos meios eletrônicos.

                (E) : Garante a empresas a obtenção de mais lucro com a recente fragilidade de nossa atenção.

                | **Resposta:** (E)
                
                Por fim, tivermos 5 questões com mais respostas corretas, das quais, não houve nenhuma de matemática, porém, em Linguagens tivemos a questão de id **2011109**, em Humanas tivemos as questões **2021090, 2010033, 2023050**, e em Ciências da Natureza tivemos a questão **2021097**.
                
                ##### 2.6.2.1 Questão mais fácil Linguagens

                O tema da velhice foi objeto de estudo de brilhantes filósofos ao longo dos tempos. Um dos melhores livros sobre o assunto foi escrito pelo pensador e orador romano Cícero: _A Arte do Envelhecimento_. Cícero nota, primeiramente, que todas as idades têm seus encantos e suas dificuldades. E depois aponta para um paradoxo da humanidade. Todos sonhamos ter uma vida longa, o que significa viver muitos anos. Quando realizamos a meta, em vez de celebrar o feito, nos atiramos a um estado de melancolia e amargura. Ler as palavras de Cícero sobre envelhecimento pode ajudar a aceitar melhor a passagem do tempo.

                NOGUEIRA, P. Saúde & Bem-Estar Antienvelhecimento. **Época**. 28 abr. 2008.

                (A) : Esclarecer que a velhice é inevitável.

                (B) : Contar fatos sobre a arte de envelhecer.

                (C) : Defender a ideia de que a velhice é desagradável.

                (D) : Influenciar o leitor para que lute contra o envelhecimento.

                (E) : Mostrar às pessoas que é possível aceitar, sem angústia, o envelhecimento.

                | Resposta: E

                ##### 2.6.2.1 Questão mais fácil Humanas

                EIGENHEER, E. M. **Lixo:** a limpeza urbana através dos tempos. Porto Alegre: Gráfica Palloti, 2009.

                **Texto II**  
                A repugnante tarefa de carregar lixo e os dejetos da casa para as praças e praias era geralmente destinada ao único escravo da família ou ao de menor status ou valor. Todas as noites, depois das dez horas, os escravos conhecidos popularmente como “tigres” levavam tubos ou barris de excremento e lixo sobre a cabeça pelas ruas do Rio.

                KARACH, M. C. **A vida dos escravos no Rio de Janeiro, 1808-1850.** Rio de Janeiro: Cia. das letras, 2000.
                (A) : Valorização do trabalho braçal.

                (B) : Reiteração das hierarquias sociais.

                (C) : Sacralização das atividades laborais.

                (D) : Superação das exclusões econômicas.

                (E) : Ressignificação das heranças religiosas.

                | Resposta: B

                ##### 2.6.2.2 Questão mais fácil Natureza

                EIGENHEER, E. M. **Lixo:** a limpeza urbana através dos tempos. Porto Alegre: Gráfica Palloti, 2009.

                **Texto II**  
                A repugnante tarefa de carregar lixo e os dejetos da casa para as praças e praias era geralmente destinada ao único escravo da família ou ao de menor status ou valor. Todas as noites, depois das dez horas, os escravos conhecidos popularmente como “tigres” levavam tubos ou barris de excremento e lixo sobre a cabeça pelas ruas do Rio.

                KARACH, M. C. **A vida dos escravos no Rio de Janeiro, 1808-1850.** Rio de Janeiro: Cia. das letras, 2000.
                (A) : Valorização do trabalho braçal.

                (B) : Reiteração das hierarquias sociais.

                (C) : Sacralização das atividades laborais.

                (D) : Superação das exclusões econômicas.

                (E) : Ressignificação das heranças religiosas.

                | Resposta: B
                """)
    
    st.divider()
    
    st.markdown("""
                ### **2.7 - Conclusão**  
                """)
    
    show_metrics(all_text_models,example_text_questions)
    
    st.markdown("""
                A partir dos experimentos conduzidos, foi possível obter uma visão detalhada sobre o desempenho de diferentes modelos de linguagem na resolução de questões de múltipla escolha, abrangendo diversos aspectos como **acurácia geral, tempo de inferência, desempenho por disciplina e comportamento frente a diferentes tipos de tarefas**. Os resultados indicam que **o número de parâmetros nem sempre é o principal fator determinante da performance**, visto que **modelos intermediários apresentaram desempenhos comparáveis a modelos robustos**, enquanto modelos especializados nem sempre demonstraram superioridade nas áreas para as quais foram ajustados.  

                Na análise do impacto do **tamanho do modelo**, observamos que, enquanto modelos menores apresentaram quedas significativas na acurácia, a diferença entre modelos intermediários e robustos foi **mínima ou inexistente**. Em alguns casos, como no **Gemma2**, a versão intermediária obteve tempos de inferência menores sem perda expressiva de precisão, sugerindo que **há um ponto de equilíbrio entre eficiência computacional e desempenho**.  

                O **tempo de inferência** demonstrou ser um fator crítico, especialmente em modelos **de reasoning e matemática**, que frequentemente apresentaram tempos muito superiores aos modelos convencionais. O **Openthinker**, por exemplo, chegou a ultrapassar **1h24min** de execução para 100 questões, enquanto os demais modelos raramente ultrapassaram **15 minutos**. Em Matemática, a **demanda computacional** foi um grande desafio, com algumas questões exigindo **até 542 segundos** para serem resolvidas, um valor muito superior ao de qualquer outra disciplina.  

                A análise **por disciplina** revelou padrões interessantes. **Ciências Humanas e Linguagens** foram as áreas de maior acurácia, com **67% e 55,8% de precisão média**, respectivamente. O desempenho superior em Ciências Humanas pode estar relacionado ao fato de que muitas questões exigem respostas mais **diretas e factuais**, reduzindo a necessidade de **interpretação subjetiva**. Já em Linguagens, a exigência de **análise contextual** e inferências mais complexas pode ter dificultado a obtenção de acertos. **Ciências da Natureza**, por sua vez, apresentou um comportamento híbrido, combinando aspectos de interpretação conceitual com questões que exigiam cálculos.  

                A **Matemática**, como esperado, foi a disciplina de maior dificuldade, tanto em acurácia quanto em tempo de inferência. Apenas **quatro modelos** conseguiram acertar mais da metade das questões matemáticas, e o tempo médio de resposta foi **mais que o dobro** das demais disciplinas. Mesmo os modelos especializados em matemática não demonstraram ganhos significativos de desempenho na área, com exceção do **Mathstral**, que superou o Mistral padrão em um fator de **4x**, evidenciando o impacto positivo do fine-tuning.  

                Quando analisamos **o desempenho por questão**, verificamos que **nenhuma questão foi completamente ignorada por todos os modelos**, o que indica um nível mínimo de compreensão em todas as áreas avaliadas. No entanto, houve uma clara distinção entre questões **fáceis e difíceis**, com algumas sendo acertadas por **mais de 15 modelos** e outras por **apenas dois modelos**. O histograma das respostas revelou uma **distribuição próxima da normalidade**, embora com algumas irregularidades, o que pode indicar que certos tipos de perguntas são consistentemente mais desafiadores para os modelos.  

                Os resultados estatísticos reforçaram essas observações. O **teste t de Student** não encontrou **diferença significativa** entre modelos intermediários e robustos, sugerindo que modelos menores podem ser opções viáveis em determinados contextos. No entanto, o **teste de equivalência (TOST)** não foi capaz de confirmar que os modelos estão dentro de um intervalo de tolerância de **±2%**, o que impede uma conclusão definitiva sobre a equivalência entre categorias de modelos.  

                Diante desses resultados, podemos concluir que **a escolha do modelo ideal deve levar em consideração múltiplos fatores além da acurácia bruta**, incluindo **tempo de inferência, consumo computacional, estabilidade das respostas e adequação às tarefas específicas**. Modelos robustos podem ser necessários para cenários que demandam **maior precisão**, mas modelos intermediários demonstraram **eficiência suficiente** para grande parte das tarefas. Além disso, a dificuldade que os modelos enfrentaram com **questões matemáticas** reforça a necessidade de avanços na capacidade de raciocínio lógico das arquiteturas atuais.  

                Por fim, a análise apresentada destaca **os desafios e limitações das arquiteturas de LLMs atuais**, ao mesmo tempo que aponta caminhos para otimizações futuras, seja na **especialização de modelos para tarefas específicas**, seja no aprimoramento do **balanço entre custo computacional e desempenho**.  
                """)