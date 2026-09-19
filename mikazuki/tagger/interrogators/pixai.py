from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from mikazuki.tagger.interrogators.base import Interrogator
from mikazuki.tagger.interrogators.onnx_gpu import create_cuda_onnx_session
from mikazuki.tagger.interrogators.tag_metadata import load_selected_tags


class PixAITaggerInterrogator(Interrogator):
    def __init__(self, name: str, repo_id: str, **kwargs) -> None:
        super().__init__(name)
        self.repo_id = repo_id
        self.kwargs = kwargs

    def download(self):
        common = {"repo_id": self.repo_id, **self.kwargs}
        return (
            Path(hf_hub_download(**common, filename="model.onnx")),
            Path(hf_hub_download(**common, filename="selected_tags.csv")),
        )

    def load(self) -> None:
        model_path, tags_path = self.download()
        self.model = create_cuda_onnx_session(model_path)
        self.tag_names, self.tag_categories = load_selected_tags(tags_path)
        print(f"Loaded {self.name} model from {model_path} with providers {self.model.get_providers()}")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        _, _, height, width = self.model.get_inputs()[0].shape
        image = image.convert("RGB")
        ratio = float(height) / max(image.size)
        new_size = tuple(int(x * ratio) for x in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (128, 128, 128))
        canvas.paste(image, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return arr.transpose(2, 0, 1)[None, ...]

    def interrogate(self, image: Image.Image) -> Dict[str, List[Tuple[str, float]]]:
        if not hasattr(self, "model") or self.model is None:
            self.load()
        img_input = self.model.get_inputs()[0]
        outputs = self.model.get_outputs()
        pred_name = outputs[2].name if len(outputs) >= 3 else outputs[-1].name
        probs = self.model.run([pred_name], {img_input.name: self._preprocess(image)})[0][0]
        result = {k: [] for k in ("rating", "general", "character", "copyright", "artist", "meta", "quality", "model")}
        category_map = {0: "general", 1: "rating", 2: "quality", 3: "meta", 4: "character", 5: "copyright", 6: "artist", 9: "rating"}
        probs = np.asarray(probs).reshape(-1)
        if len(probs) != len(self.tag_names):
            raise RuntimeError(
                f"{self.name} output/tag count mismatch: "
                f"model returned {len(probs)} scores, "
                f"but selected_tags.csv contains {len(self.tag_names)} tags"
            )

        for name, category_id, score in zip(
            self.tag_names,
            self.tag_categories,
            probs,
        ):
            category = category_map.get(category_id, "general")
            result[category].append((name, float(score)))
        return result
