from .classifier import StiffnessClassifier, Trainer


def build_classifier_model(encoder, model_cfg):
    model = StiffnessClassifier(encoder, model_cfg)
    trainer = Trainer(model)
    return model, trainer