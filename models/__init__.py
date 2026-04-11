# Model registry for all CNN variants
from .resnet_baseline import ResNet18Baseline
from .resnet_dilated import ResNet18Dilated
from .resnet_deformable import ResNet18Deformable
from .resnet_depthwise import ResNet18Depthwise
from .resnet_hybrid import ResNet18Hybrid

MODEL_REGISTRY = {
    'baseline': ResNet18Baseline,
    'dilated': ResNet18Dilated,
    'deformable': ResNet18Deformable,
    'depthwise': ResNet18Depthwise,
    'hybrid': ResNet18Hybrid,
}
