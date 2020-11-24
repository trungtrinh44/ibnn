from .utils import *
from .resnet import *
from .vgg import *

def get_model_from_config(config):
    model_name = config.get('model_name', config.get('model'))
    if model_name ==  'StoWideResNet28x10':
        return StoWideResNet28x10(config['num_classes'], config['n_components'], 1.0, 0.3)
    if model_name ==  'DetWideResNet28x10':
        return DetWideResNet28x10(config['num_classes'], config.get('dropout', 0))
    if model_name ==  'StoVGG16':
        return StoVGG16(config['num_classes'], config['n_components'], 1.0, 0.3, (1.0, 0.5), (0.05, 0.02))
    if model_name ==  'DetVGG16':
        return DetVGG16(config['num_classes'])
    if model_name ==  'BayesianVGG16':
        return BayesianVGG16(config['num_classes'], 1.0, 0.3)
    if model_name ==  'BayesianWideResNet28x10':
        return BayesianWideResNet28x10(config['num_classes'], 1.0, 0.3)
