from .Graph_mae import MAE2


def build_graph_model(cfg):
    # model_cfg = args.cfg
    model = MAE2(
        cfg = cfg,
    )
    return model
