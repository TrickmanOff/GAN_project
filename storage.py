"""Знает об устройстве директорий и именах файлов"""
import torch

import os

from device import get_local_device


class ModelDir:
    def __init__(self, model_dirpath: str,
                 checkpoint_filename: str = 'training_checkpoint',
                 model_state_filename: str = 'model_state') -> None:
        self.model_dirpath = model_dirpath
        self.checkpoint_filepath = os.path.join(model_dirpath, checkpoint_filename)
        self.model_state_filepath = os.path.join(model_dirpath, model_state_filename)

    def get_checkpoint_state(self) -> dict | None:
        if not os.path.exists(self.checkpoint_filepath):
            return None
        checkpoint = torch.load(self.checkpoint_filepath, map_location=get_local_device())
        return checkpoint

    def save_checkpoint_state(self, checkpoint: dict) -> None:
        torch.save(checkpoint, self.checkpoint_filepath)


class ExperimentsStorage:
    def __init__(self, experiments_dir: str = './experiments',
                 checkpoint_filename: str = 'training_checkpoint',
                 model_state_filename: str = 'model_state') -> None:
        self.experiments_dir = experiments_dir
        self.checkpoint_filename = checkpoint_filename
        self.model_state_filename = model_state_filename
        if not os.path.exists(self.experiments_dir):
            os.mkdir(self.experiments_dir)

    def get_model_dir(self, model_name: str) -> ModelDir:
        model_dirpath = os.path.join(self.experiments_dir, model_name)
        if not os.path.exists(model_dirpath):
            os.mkdir(model_dirpath)
        return ModelDir(model_dirpath=model_dirpath, checkpoint_filename=self.checkpoint_filename,
                        model_state_filename=self.model_state_filename)
