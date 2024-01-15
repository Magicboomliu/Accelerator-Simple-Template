from transformers import PretrainedConfig

class ConNetConfig(PretrainedConfig):
    model_type = "convnet"

    def __init__(
        self,
        num_classes=10,
        **kwargs,
    ):
        self.num_classes = num_classes
        super().__init__(**kwargs)


if __name__=="__main__":
    convnet_config = ConNetConfig(num_classes=10)
    convnet_config.save_pretrained("custom-convnet")
    
    pass