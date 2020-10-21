from .utils import *
from .resnet import *
from .vgg import *

def get_model_from_config(config):
    if config['model_name'] == 'StoWideResNet28x10':
        return StoWideResNet28x10(config['num_classes'], config['n_components'], config['prior_mean'], config['prior_std'])
    if config['model_name'] == 'DetWideResNet28x10':
        return DetWideResNet28x10(config['num_classes'], config['dropout'])
    if config['model_name'] == 'StoVGG16':
        return StoVGG16(config['num_classes'], config['n_components'], config['prior_mean'], config['prior_std'], config.get('posterior_mean_init', (1.0, 0.5)), config.get('posterior_std_init', (0.05, 0.02)))
    if config['model_name'] == 'DetVGG16':
        return DetVGG16(config['num_classes'])
    if config['model_name'] == 'BayesianVGG16':
        return BayesianVGG16(config['num_classes'], config['prior_mean'], config['prior_std'])
    if config['model_name'] == 'BayesianWideResNet28x10':
        return BayesianWideResNet28x10(config['num_classes'], config['prior_mean'], config['prior_std'])
