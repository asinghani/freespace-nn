import os
import yaml

# General
random_seed = 42

class Config(object):
    def __init__(self):
        # Config options declared in this function

        i = 11

        self.weights_save_location = "/hdd/models/floorseg/floornet_{}".format(i)

        self.training_log_location = os.path.join(self.weights_save_location, "training_log.csv")
        self.config_save_location = os.path.join(self.weights_save_location, "train_config.yaml")
        self.model_vis_save_location = os.path.join(self.weights_save_location, "model.png")

        # Dataset Params
        self.train_test_split = 0.1
        self.augment_ratio = 0.5

        self.images_location = "/hdd/datasets/floorseg/raw"
        self.labels_location = "/hdd/datasets/floorseg/labels"

        self.image_size = (224, 224)

        self.input_shape = (self.image_size[0], self.image_size[1], 3) # should height and width be reversed ??

        # Hyperparams
        self.samples_per_epoch = 200
        self.total_epoch = 1000
        self.batch_size = 16
        self.test_batch_size = 16

        self.learning_rate = 3.5e-4
        self.dropout = 0.4
        self.l2_constant = 1e-3

        self.mobilenet_alpha = 1.0
        self.include_softmax = True

        self.init_weights = None #"/home/anish/mobilenet_weights.h5" # path to weights

    def serialize(self):
        return yaml.dump(self.__dict__)
