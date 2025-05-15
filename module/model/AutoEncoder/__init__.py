from .ImageAE import ImageAutoEncoder, Trainer

def build_model(model_cfg, comm_cfg=None):
    model = ImageAutoEncoder(model_cfg, comm_cfg)
    trainer = Trainer(model)

    return model, trainer
