def load_t5(input_shape, num_classes):
    from ..datasets.utils import configure_cache_env
    from transformers import T5ForConditionalGeneration

    configure_cache_env()
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    return model
