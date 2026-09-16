import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from mikazuki.tagger.interrogators.base import Interrogator
from mikazuki.tagger.interrogators.onnx_gpu import create_cuda_onnx_session


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    artist: list[np.int64]
    character: list[np.int64]
    copyright: list[np.int64]
    meta: list[np.int64]
    quality: list[np.int64]
    model: list[np.int64]


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    new_size = max(width, height)
    new_image = Image.new(image.mode, (new_size, new_size), (255, 255, 255))
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_image.paste(image, paste_position)
    return new_image


def get_tags(probs, labels: LabelData):
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

    if len(labels.rating) > 0:
        valid_indices = labels.rating[labels.rating < len(probs)]
        if len(valid_indices) > 0:
            rating_probs = probs[valid_indices]
            if len(rating_probs) > 0:
                rating_idx_local = np.argmax(rating_probs)
                rating_idx_global = valid_indices[rating_idx_local]
                if rating_idx_global < len(labels.names) and labels.names[rating_idx_global] is not None:
                    result["rating"].append((labels.names[rating_idx_global], float(rating_probs[rating_idx_local])))

    if len(labels.quality) > 0:
        valid_indices = labels.quality[labels.quality < len(probs)]
        if len(valid_indices) > 0:
            quality_probs = probs[valid_indices]
            if len(quality_probs) > 0:
                quality_idx_local = np.argmax(quality_probs)
                quality_idx_global = valid_indices[quality_idx_local]
                if quality_idx_global < len(labels.names) and labels.names[quality_idx_global] is not None:
                    result["quality"].append((labels.names[quality_idx_global], float(quality_probs[quality_idx_local])))

    category_map = {
        "general": labels.general,
        "character": labels.character,
        "copyright": labels.copyright,
        "artist": labels.artist,
        "meta": labels.meta,
        "model": labels.model,
    }
    for category, indices in category_map.items():
        if len(indices) > 0:
            valid_indices = indices[indices < len(probs)]
            if len(valid_indices) > 0:
                category_probs = probs[valid_indices]
                for idx_local, idx_global in enumerate(valid_indices):
                    if idx_global < len(labels.names) and labels.names[idx_global] is not None:
                        result[category].append((labels.names[idx_global], float(category_probs[idx_local])))

    for key in result:
        result[key] = sorted(result[key], key=lambda item: item[1], reverse=True)
    return result


class CLTaggerInterrogator(Interrogator):
    def __init__(
            self,
            name: str,
            model_path='model.onnx',
            tag_mapping_path='tag_mapping.json',
            **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tag_mapping_path = tag_mapping_path
        self.kwargs = kwargs

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")
        model_path = Path(hf_hub_download(**self.kwargs, filename=self.model_path))
        tag_mapping_path = Path(hf_hub_download(**self.kwargs, filename=self.tag_mapping_path))
        return model_path, tag_mapping_path

    def load(self) -> None:
        model_path, tag_mapping_path = self.download()
        self.model = create_cuda_onnx_session(model_path)
        print(f'Loaded {self.name} model from {model_path} with providers {self.model.get_providers()}')
        self.tags = self.load_tag_mapping(tag_mapping_path)

    def load_tag_mapping(self, mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as file:
            tag_mapping_data = json.load(file)

        if isinstance(tag_mapping_data, dict) and "idx_to_tag" in tag_mapping_data:
            idx_to_tag = {int(key): value for key, value in tag_mapping_data["idx_to_tag"].items()}
            tag_to_category = tag_mapping_data["tag_to_category"]
        elif isinstance(tag_mapping_data, dict):
            try:
                data_int_keys = {int(key): value for key, value in tag_mapping_data.items()}
                idx_to_tag = {idx: data['tag'] for idx, data in data_int_keys.items()}
                tag_to_category = {data['tag']: data['category'] for data in data_int_keys.values()}
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Unsupported tag mapping format (dict): {exc}. Expected int keys with 'tag' and 'category'."
                ) from exc
        else:
            raise ValueError("Unsupported tag mapping format: Expected a dictionary.")

        names = [None] * (max(idx_to_tag.keys()) + 1)
        rating, general, artist, character, copyright, meta, quality, model_name = [], [], [], [], [], [], [], []
        for idx, tag in idx_to_tag.items():
            if idx >= len(names):
                names.extend([None] * (idx - len(names) + 1))
            names[idx] = tag
            category = tag_to_category.get(tag, 'Unknown')
            if category == 'Rating':
                rating.append(idx)
            elif category == 'General':
                general.append(idx)
            elif category == 'Artist':
                artist.append(idx)
            elif category == 'Character':
                character.append(idx)
            elif category == 'Copyright':
                copyright.append(idx)
            elif category == 'Meta':
                meta.append(idx)
            elif category == 'Quality':
                quality.append(idx)
            elif category == 'Model':
                model_name.append(idx)

        labels = LabelData(
            names=names,
            rating=np.array(rating, dtype=np.int64),
            general=np.array(general, dtype=np.int64),
            artist=np.array(artist, dtype=np.int64),
            character=np.array(character, dtype=np.int64),
            copyright=np.array(copyright, dtype=np.int64),
            meta=np.array(meta, dtype=np.int64),
            quality=np.array(quality, dtype=np.int64),
            model=np.array(model_name, dtype=np.int64),
        )
        return labels, idx_to_tag, tag_to_category

    def preprocess_image(self, image: Image.Image, target_size=(448, 448)):
        image = pil_ensure_rgb(image)
        image = pil_pad_square(image)
        image_resized = image.resize(target_size, Image.BICUBIC)
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = img_array[::-1, :, :]
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        img_array = np.expand_dims(img_array, axis=0)
        return image, img_array

    def interrogate(self, image: Image) -> dict[str, list]:
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        _, input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.astype(np.float32)

        outputs = self.model.run([output_name], {input_name: input_tensor})[0]
        if np.isnan(outputs).any() or np.isinf(outputs).any():
            print("Warning: NaN or Inf detected in model output. Clamping...")
            outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)

        probs = 1 / (1 + np.exp(-np.clip(outputs[0], -30, 30)))
        predictions = get_tags(probs, self.tags[0])
        print(predictions)
        return predictions
