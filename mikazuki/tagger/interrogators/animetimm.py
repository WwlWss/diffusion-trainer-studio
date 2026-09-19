import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

from mikazuki.tagger.interrogators.base import Interrogator
from mikazuki.tagger.interrogators.onnx_gpu import create_cuda_onnx_session
from mikazuki.tagger.interrogators.tag_metadata import load_selected_tags


class AnimeTimmInterrogator(Interrogator):
    """ONNX interrogator for AnimeTimm dbv4-full models."""

    def __init__(self, name: str, repo_id: str, **kwargs) -> None:
        super().__init__(name)
        self.repo_id = repo_id
        self.kwargs = kwargs

    def _download_file(self, filename: str) -> Path:
        common = {"repo_id": self.repo_id, **self.kwargs}

        # A cached model must remain usable even if its upstream repository later
        # becomes gated or the machine is temporarily offline. Try the local HF
        # cache without a network request first, then fall back to the normal
        # authenticated download path only when the file is missing.
        try:
            return Path(hf_hub_download(**common, filename=filename, local_files_only=True))
        except LocalEntryNotFoundError:
            pass

        try:
            return Path(hf_hub_download(**common, filename=filename))
        except GatedRepoError as exc:
            raise RuntimeError(
                f"AnimeTimm model '{self.repo_id}' is gated on Hugging Face. "
                "Accept the model's access terms in your browser and authenticate "
                "Hugging Face locally (or provide HF_TOKEN), then retry."
            ) from exc

    def download(self):
        return (
            self._download_file("model.onnx"),
            self._download_file("selected_tags.csv"),
            self._download_file("preprocess.json"),
        )

    def load(self) -> None:
        model_path, tags_path, preprocess_path = self.download()
        # The shared helper requires CUDA to be available and verifies ORT's
        # actual graph assignment. CPU remains available only for the small
        # shape/control nodes that ORT deliberately keeps there; substantial
        # tagger compute must be assigned to CUDA or session creation fails.
        self.model = create_cuda_onnx_session(model_path)

        self.tag_names, self.tag_categories = load_selected_tags(tags_path)
        self.preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
        print(f"Loaded {self.name} model from {model_path} with providers {self.model.get_providers()}")

    @staticmethod
    def _resize_shorter_side(image: Image.Image, size):
        if isinstance(size, (list, tuple)):
            h, w = int(size[0]), int(size[1])
            return image.resize((w, h), Image.Resampling.BICUBIC)
        w, h = image.size
        scale = float(size) / min(w, h)
        return image.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)

    @staticmethod
    def _center_crop(image: Image.Image, size):
        if isinstance(size, int):
            crop_h = crop_w = size
        else:
            crop_h, crop_w = int(size[0]), int(size[1])
        w, h = image.size
        left = max((w - crop_w) // 2, 0)
        top = max((h - crop_h) // 2, 0)
        return image.crop((left, top, left + crop_w, top + crop_h))

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        steps = self.preprocess["test"]
        image = image.convert("RGB")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for step in steps:
            step_type = step["type"]
            if step_type == "pad_to_size":
                pad_h, pad_w = step["size"]
                w, h = image.size
                if h < pad_h or w < pad_w:
                    canvas = Image.new("RGB", (max(w, pad_w), max(h, pad_h)), "white")
                    canvas.paste(image, ((canvas.width - w) // 2, (canvas.height - h) // 2))
                    image = canvas
            elif step_type == "resize":
                image = self._resize_shorter_side(image, step["size"])
            elif step_type == "center_crop":
                image = self._center_crop(image, step["size"])
            elif step_type == "normalize":
                mean = step["mean"]
                std = step["std"]
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
        return arr.transpose(2, 0, 1)[None, ...]

    def interrogate(self, image: Image.Image) -> Dict[str, List[Tuple[str, float]]]:
        if not hasattr(self, "model") or self.model is None:
            self.load()
        img_input = self.model.get_inputs()[0]
        outputs = self.model.run(None, {img_input.name: self._preprocess(image)})
        logits = max(outputs, key=lambda x: x.shape[-1])[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
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
