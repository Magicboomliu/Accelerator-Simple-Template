from convnet.configuration_convnet import ConNetConfig
from convnet.modeling_convnet import ConvNetModel
import torch



# model = ConvNetModel.from_pretrained("Zihua-Liu-CVer/custom-convnet")

ConNetConfig.register_for_auto_class()
ConvNetModel.register_for_auto_class("AutoModel")



convet_config = ConNetConfig(num_classes=10)
convnet = ConvNetModel(convet_config)

ckpt = torch.load("saved_path/ckpt_5.pt")

convnet.model.load_state_dict(ckpt['model_state'])
print("Loaded Successfully")

convnet.push_to_hub('custom-convnet')
# model = ConvNetModel.from_pretrained("Zihua-Liu-CVer/custom-convnet")
