import abc
import logging
import pickle as pkl
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import tensorflow as tf

from configs import ModelConfBase

log = logging.getLogger(name=__name__)


class ModelBase:
    """
        Base class for keras models routine.
    """

    MODEL_CONFIG_KEY = 'model_config'
    WEIGHTS_KEY = 'weights'

    def __init__(self, config: ModelConfBase, mode: Optional[str] = None):
        self.config = config
        self.mode = mode
        self.keras_model = None

    @classmethod
    def new(cls, config: ModelConfBase, mode: Optional[str] = None):
        """
            Creates new model with random weights.
        """
        log.info("No weights")
        object = cls(config=config, mode=mode)
        object.build()  # Construct the model
        return object

    @classmethod
    def load(cls, source: Union[Path, str], mode: Optional[str] = None):
        """
            Loads model from file.
            Args:
                source: path to pkl weights file
            Returns:
                ModelBase object
        """
        log.info(f"Load weights from {source}...")

        with source.open('rb') as file:
            model_dump = pkl.load(file)

        # Get config and weights from file
        config = model_dump[cls.MODEL_CONFIG_KEY]
        weights = model_dump[cls.WEIGHTS_KEY]

        object = cls(config=config, mode=mode)
        object.build()  # Construct the model
        object.set_weights(weights=weights)  # Load weights

        log.info(f"Weights from {source} are loaded successfully.")
        return object

    def set_weights(self, weights: dict, missing_names_ok: bool = False) -> None:
        """
            Set weights from dictionary.
        """
        for layer in self.keras_model.layers:
            if layer.name in weights:
                layer.set_weights(weights[layer.name])
            elif missing_names_ok or not layer.weights:
                continue
            else:
                raise ValueError(f"Can't set weights for layer: {layer.name}")

    def save_pickled(self, model_path: Path):
        """
            Save model in .pkl format.
        """
        data = {
            self.MODEL_CONFIG_KEY: self.config,
            self.WEIGHTS_KEY: {layer.name: layer.get_weights() for layer in self.keras_model.layers},
        }

        with model_path.open('wb') as file:
            pkl.dump(data, file)

        log.info(f"Weights saved successfully: {model_path}")

    @abc.abstractmethod
    def _build(self) -> tf.keras.Model:
        """
            Method implements concrete model construction
        """
        raise NotImplementedError('Method must be implemented in child class!')

    def build(self):
        self.keras_model = self._build()

    def predict(self, data: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """
            Get predictions on a batch of images
            Args:
                data: numpy array of u_int8 data in format (batch, model_height, model_width, channels)
            Returns:
                model prediction
        """
        input_data = self.normalize(data=data)
        return self.keras_model.predict(input_data)

    @staticmethod
    @abc.abstractmethod
    def normalize(data):
        """
            Method implements data normalization function
        """
        raise NotImplementedError('Method must be implemented in child class!')

    @staticmethod
    @abc.abstractmethod
    def restore_normalized(data):
        """
            Method implements inverted 'self.normalize' function (revert normalization back to original values)
        """
        raise NotImplementedError('Method must be implemented in child class!')
