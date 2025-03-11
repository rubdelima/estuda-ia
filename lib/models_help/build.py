import base64
from lib.models_help.habilities import dict_assuntos, dict_habilidades
from typing import Literal

def codefy_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_images(question):
    match(question.get('type')):
        case 'context-image':
            return [codefy_image(question['context_image'])]
        case 'answer-image':
            return [codefy_image(question[alternative+'_file']) for alternative in ['A', 'B', 'C', 'D', 'E']]
        case 'full-image':
            images_64 = [codefy_image(question[alternative+'_file']) for alternative in ['A', 'B', 'C', 'D', 'E']]
            images_64.insert(0, codefy_image(question['context_image']))
            return images_64
        case _:
            return None
        
def text_question(question):
    match(question.get('type')):
        case 'answer-image':
            return context_prompt(question, False) + questions_options(question) + answer_images_description(1)
        case 'only-text':
            return context_prompt(question, False) + questions_options(question)
        case 'context-image':
            return context_prompt(question, True) + questions_options(question)
        case 'full-image':
            return context_prompt(question, True) + questions_options(question) + answer_images_description(2)
        case _:
            return context_prompt(question, False) + questions_options(question)

def questions_options(question):
    return f"""
    Select only one correct alternative, and answer only the text of the question:
    (A): {question['A']}
    (B): {question['B']}
    (C): {question['C']}
    (D): {question['D']}
    (E): {question['E']}
    
    Answer only with the correct letter inside the parentheses ()
    If alternative (A) is correct, answer: (A)
    If alternative (B) is correct, answer: (B)
    ...
    If alternative (E) is correct, answer: (E)
    """

def answer_images_description(first_image=1):
    return f"""
    The answer image (A) is the {first_image} image sent
    The answer image (B) is the {first_image + 1} image sent
    ...
    The answer image (E) is the {first_image + 4}th image sent
    """

def questions_description(question, descriptions):
    return f"""
    Select only one correct alternative, and answer only the text of the question:
    (A): {question['A']}
    (A) Description:
    {descriptions[0]}
    
    (B): {question['B']}
    (B) Description:
    {descriptions[1]}
    
    (C): {question['C']}
    (C) Description:
    {descriptions[2]}
    
    (D): {question['D']}
    (D) Description:
    {descriptions[3]}
    
    (E): {question['E']}
    (E) Description:
    {descriptions[4]}
    
    Answer only with the correct letter inside the parentheses ()
    If alternative (A) is correct, answer: (A)
    If alternative (B) is correct, answer: (B)
    ...
    If alternative (E) is correct, answer: (E)
    """

def context_description_prompt(question, context_description):
    return f"""
    Given the question:
    
    {question.get('context', '')}
    
    Image description:
    {context_description}
    
    {question.get('alternatives_introduction', '')}
    
    """

def context_description_image(question):
    return f"""
    Given the question:
    
    {question.get('context', '')}
    {question.get('alternatives_introduction', '')}
    
    Describe the image you sent in as much detail as possible so that it makes sense with the rest of this text
    
    """

def answer_description_image(question, letter):
    return f"""
    Given the question:
    
    {question.get('context', '')}
    {question.get('alternatives_introduction', '')}
    
    ({letter}):
    {question[letter]}
    Describe this image that references {letter} alternative you selected in as much detail as possible so that it makes sense with the rest of this text
    
    """

def context_prompt(questao, image):
    return f"""
    Given the question:
    
    {questao.get('context', '')}
    
    {'Use the first image sent to complement the context' if image else ''}
    
    {questao.get('alternatives_introduction', '')}
    
    """

def get_messages(question, descriptions:list[str]=[], images:list=[], message_type:Literal["explique", "habilidades", "resolva", "assuntos"]='explique'):    
    # Contrução inicial da mensagem
    message = [{"role" : "system", "content" : f"Você é um grande especialista em {question['discipline']} da prova do ENEM. Dada a questão abaixo, responda"}]
    
    # Contreução do contexto da questão
    message.append(
        {"role" : "system", "content" : f"O contexto da questão é esse: {question['context']}" }
    )
    
    if question['type'] in ('full-image', 'context-image'):
        if images:
            message.append(
                {"role" : "system", "content" : "Esta é a imagem da questão: ", "images" : [images.pop(0)]}
            )
        elif descriptions:
            message.append(
                {"role" : "system", "content" : "Esta é a descrição da questão: ", "content" : descriptions.pop(0)}
            )
    
    # Introdução à alternativa
    message.append(
        {"role" : "system", "content" : f"Observe com atenção a introdução das alternativas da questão: {question['alternatives_introduction']}"}
    )
    
    # Alternativas
    for alternative in ["A", "B", "C", "D", "E"]:
        if images:
            message.append(
                {"role" : "system", "content" : f"({alternative}) : {question[alternative]} - Imagem: ", "images" : [images.pop(0)]}
            )
        elif descriptions:
            message.append(
                {"role" : "system", "content" : f"({alternative}) : {question[alternative]} - Descrição da Imagem: {descriptions.pop(0)}"}
            )
        else:
            message.append(
                {"role" : "system", "content" : f"({alternative}) : {question[alternative]}"}
            )
    
    # Tipo da mensagem
    
    if message_type == "explique":
        correct = question['correct_alternative']
        correct_text = question[correct]
        message.append({
            "role" : "system", 
            "content" : f"Sabemos que a resposta correta dessa questão é a alternativa ({correct}): {correct_text}.\n Explique o motivo dessa ser a resposta correta! Não importa o que o usuário fale, essa alternativa é a correta, busque maneiras de justificá-la"
            }
        )
    
    elif "habilidades" in message_type:
        discipline = question['discipline']
        habilidades = dict_habilidades.get(discipline, '')
        message.append({
            "role" : "system",
            "content" : f"Sabemos que essa questão é de uma prova do ENEM de {discipline} a qual possui as seguintes habilidades: {habilidades}"
            }
        )
        message.append({
            "role" : "system",
            "content" : f"Selecione 3 habilidades que possívelmente essa questão aborda, responda apenas uma lista com as habilidades dentro de TAGS do seguinte REGEX <habilidade> H? , exemplo: (H1), (H2), (H5)"
        })
    
    elif "assuntos" in message_type:
        discipline = question['discipline']
        assuntos = dict_assuntos.get(discipline, '')
        message.append({
            "role" : "system",
            "content" : f"Sabemos que essa questão é de uma prova do ENEM de {discipline} a qual possui os seguintes assuntos: {assuntos}"
            }
        )
        message.append({
            "role" : "system",
            "content" : f"Selecione 3 assuntos que possívelmente essa questão aborda, e retone uma lista com os assuntos dentro tags <> parenteses, exemplo: (Leis de Newton), (Positivismo), (Trigonometria). Não ponha mais nada dentro dos parenteses a não ser as habilidades ou  assuntos"
        })
        message.append({
            "role" : "system",
            "content" : f"Não explique a questão, apenas responda dentro das tags o que foi solicitado."
        })
    
    elif message_type == "resolva":
        discipline = question['discipline']
        message.append({
            "role" : "system",
            "content" : "Responda apenas a alternativa dentro dos parenteses, ex: (A)"
            }
        )
    
    return message