from tensorflow.keras.models import load_model
from config import MODEL_PATHS

_model_cache = {}

def get_model(model_name):
    if model_name not in _model_cache:
        path = MODEL_PATHS[model_name]
        _model_cache[model_name] = load_model(path, compile=False)
    return _model_cache[model_name]