import ollama

def list_avaliable_models():
    models = {}
    for model in ollama.list().models:
        models[model.model] = {
            "model_name": model.model,
            "size" : model.size/1e9,
            "parameters" : model.details.parameter_size
        }
    return models

def check_ollama():
    try:
        ollama.list()
        return True
    except Exception as e:
        return False

def check_model(model):
    try:
        ollama.show(model)
        print(f"✅ {model} disponível")
        return True
    except:
        print(f"❌ {model} indisponível")
        return False

def check_models(models):
    for model in models:
        if check_model(model):
            yield model
