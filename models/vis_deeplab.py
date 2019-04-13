from .deeplab_mobilenet import DeepLabV3_MobileNetV2
from ..config import Config

from tensorflow.keras.utils import plot_model

config = Config()
config.init_weights = None
model = DeepLabV3_MobileNetV2(config, tx2_gpu=False)

plot_model(model, to_file="/hdd/datasets/temp/model.png", show_shapes=True)
