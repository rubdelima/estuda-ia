import base64

def codefy_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_images(question):
    match(question.get('type')):
        case 'context':
            return [codefy_image(question['context_image'])]
        case _:
            return None
        
def text_question(question):
    match(question.get('type')):
        case 'only_text':
            return only_text_question(question)
        case 'context':
            return context_question(question)
        case _:
            return only_text_question(question)

def questions_options(questao):
    return f"""
    Select only one correct alternative, and answer only the text of the question:
    (A): {questao['A']}
    (B): {questao['B']}
    (C): {questao['C']}
    (D): {questao['D']}
    (E): {questao['E']}
    
    Answer only with the correct letter inside the parentheses ()
    If alternative (A) is correct, answer: (A)
    If alternative (B) is correct, answer: (B)
    ...
    If alternative (E) is correct, answer: (E)
    """

def only_text_question(questao):
    return f"""
    Observe a pergunta:
    {questao.get('context', '')}
    
    {questao.get('alternatives_introduction')}
    
    {questions_options(questao)}
    """

def context_question(questao):
    return f"""
    Given the question:
    
    {questao.get('context', '')}
    
    Use the image sent to complement the context
    
    {questao.get('alternatives_introduction', '')}
    
    {questions_options(questao)}
    """