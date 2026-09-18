# from https://github.com/toriato/stable-diffusion-webui-wd14-tagger
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

from mikazuki.tagger.interrogators.base import Interrogator
from mikazuki.tagger.interrogators.onnx_gpu import create_cuda_onnx_session
from mikazuki.tagger import dbimutils


class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
            self,
            name: str,
            model_path='model.onnx',
            tags_path='selected_tags.csv',
            **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")
        model_path = Path(hf_hub_download(**self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(**self.kwargs, filename=self.tags_path))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()
        self.model = create_cuda_onnx_session(model_path)
        print(f'Loaded {self.name} model from {model_path} with providers {self.model.get_providers()}')
        self.tags = pd.read_csv(tags_path)

    def interrogate(
            self,
            image: Image
    ) -> Dict[str, List[Tuple[str, float]]]:
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        _, height, _, _ = self.model.get_inputs()[0].shape

        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)
        image = image[:, :, ::-1]
        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]
        ratings = dict(tags[:4].values)
        tags = dict(tags[4:].values)

        result = {
            "rating": [],
            "general": [],
            "character": [],
            "copyright": [],
            "artist": [],
            "meta": [],
            "quality": [],
            "model": []
        }

        for tag, conf in ratings.items():
            result["rating"].append((tag, conf))
        for tag, conf in tags.items():
            result["general"].append((tag, conf))
        return result
