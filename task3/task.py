import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model
from PIL import Image
import os
from cutout import Cutout
from task2.densenet3 import DenseNet3
from utils import get_concat_h_multi_resize, get_concat_v_multi_resize