from models.efficientnet import EfficientNetV2

def get_model(config, **kwargs):
    name = config['model']
    if name == "efficientnet":
        return EfficientNetV2(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")