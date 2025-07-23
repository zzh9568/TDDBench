'''
This is a configuration file for converting the model used in TDDBench to the Hugging Face format.
'''
import sys
from transformers import PreTrainedModel, PretrainedConfig
sys.path.insert(0, "./basic/")
from models import MLP
from wide_resnet import WideResNet
from resnet import ResNet, BasicBlock, Bottleneck
from mobilenet import MobileNetV2
from vgg import VGG

class MLPConfig(PretrainedConfig):
    model_type = "mlp"  # 必须定义，用于标识模型类型
    
    def __init__(
        self,
        tdd_label=[0],
        in_shape=3,
        hiddens=[512, 256, 128, 64],
        num_classes=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tdd_label = list(tdd_label)
        self.in_shape = in_shape
        self.hiddens = hiddens
        self.num_classes = num_classes

class MLPHFModel(PreTrainedModel):
    config_class = MLPConfig  # 关联配置类
    
    def __init__(self, config):
        super().__init__(config)
        # 从config初始化参数
        self.model = MLP(
            in_shape=config.in_shape,
            hiddens=config.hiddens,
            num_classes=config.num_classes,
        )

    def forward(self, x, **kwargs):
        return self.model(x)
    
class WRNConfig(PretrainedConfig):
    model_type = "WideResNet"  # 必须定义，用于标识模型类型
    
    def __init__(
        self,
        tdd_label=[0],
        in_shape=3,
        num_classes=10,
        depth=28,
        width=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tdd_label = list(tdd_label)
        self.in_shape = in_shape
        self.num_classes = num_classes
        self.depth = depth
        self.width = width

class WRNHFModel(PreTrainedModel):
    config_class = WRNConfig  # 关联配置类
    
    def __init__(self, config):
        super().__init__(config)
        # 从config初始化参数
        self.model = WideResNet(
            nin=config.in_shape,
            nclass=config.num_classes,
            depth=config.depth, 
            width=config.width,
        )

    def forward(self, x, **kwargs):
        return self.model(x)
