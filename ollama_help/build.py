import base64

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

