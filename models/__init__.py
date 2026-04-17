# Model registry for all model variants
from .resnet_baseline import ResNet18Baseline
from .resnet_dilated import ResNet18Dilated
from .resnet_deformable import ResNet18Deformable
from .resnet_depthwise import ResNet18Depthwise
from .resnet_hybrid import ResNet18Hybrid
from .resnet50_baseline import ResNet50Baseline
from .resnet_frozen import ResNet18Frozen
from .vit_model import ViTForFlowerRecognition
from .vit_model_deep import ViTDeepForFlowerRecognition

MODEL_REGISTRY = {
    'baseline': ResNet18Baseline,
    'dilated': ResNet18Dilated,
    'deformable': ResNet18Deformable,
    'depthwise': ResNet18Depthwise,
    'hybrid': ResNet18Hybrid,
    'resnet50': ResNet50Baseline,
    'frozen': ResNet18Frozen,
    'vpt_shallow': ViTForFlowerRecognition,
    'vpt_deep': ViTDeepForFlowerRecognition,
}
