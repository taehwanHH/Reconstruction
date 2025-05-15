from .resnet20 import ResNet20Encoder, ResNet20Decoder
from .resnet56 import ResNet56Encoder, ResNet56Decoder


# Mapping from backbone identifier to Encoder/Decoder classes
_ENCODER_DECODER_MAP = {
    'resnet20': (ResNet20Encoder, ResNet20Decoder),
    'resnet56': (ResNet56Encoder, ResNet56Decoder),
    # Add more mappings as needed
}

def setup_enc_dec(backbone, latent_dim, drop_p=0.2):
    """
    Given a backbone identifier or pretrained model instance, returns
    (encoder, decoder) instances configured with latent_dim and drop_p.
    """
    # Determine mapping key
    if isinstance(backbone, str):
        key = backbone.lower()
    else:
        # Infer key from class name of the backbone model
        cls_name = backbone.__class__.__name__.lower()
        if 'resnet20' in cls_name:
            key = 'resnet20'
        elif 'resnet56' in cls_name:
            key = 'resnet56'
        else:
            raise ValueError(f"Unsupported backbone instance: {cls_name}")

    if key not in _ENCODER_DECODER_MAP:
        raise ValueError(f"No encoder/decoder registered for key '{key}'")

    EncoderClass, DecoderClass = _ENCODER_DECODER_MAP[key]
    encoder = EncoderClass(latent_dim=latent_dim)
    decoder = DecoderClass(latent_dim=latent_dim, drop_p=drop_p)
    return encoder, decoder