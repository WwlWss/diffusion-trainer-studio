import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

from mikazuki.tagger.interrogators.base import Interrogator
from mikazuki.tagger.interrogators.onnx_gpu import create_cuda_onnx_session


class DanbooruTagQueryInterrogator(Interrogator):
    def __init__(self, name: str, variant: str, repo_id: str = "realphongha/danbooru-tag-query") -> None:
        super().__init__(name)
        self.variant = variant
        self.repo_id = repo_id

    def _resolve_variant(self) -> str:
        files = HfApi().list_repo_files(self.repo_id, repo_type="model")
        variants = sorted({
            path.split("/")[1]
            for path in files
            if path.startswith("models/") and path.count("/") >= 2
        })
        if self.variant in variants:
            return self.variant
        needle = self.variant.lower().replace("/", "")
        matches = [v for v in variants if needle in v.lower().replace("/", "")]
        if not matches:
            raise FileNotFoundError(
                f"No DanbooruTagQuery variant matching '{self.variant}' found in {self.repo_id}. "
                f"Available variants: {', '.join(variants)}"
            )
        return matches[0]

    def download(self):
        variant = self._resolve_variant()
        prefix = f"models/{variant}"
        model_path = Path(hf_hub_download(repo_id=self.repo_id, filename=f"{prefix}/model.onnx"))
        sidecars = {}
        for filename in ("config.json", "tag_to_id.json", "tag_category.json"):
            sidecars[filename] = Path(hf_hub_download(repo_id=self.repo_id, filename=f"{prefix}/{filename}"))
        return model_path, sidecars

    def load(self) -> None:
        model_path, sidecars = self.download()
        self.model = create_cuda_onnx_session(model_path)
        self.config = json.loads(sidecars["config.json"].read_text(encoding="utf-8"))
        self.tag_to_id = json.loads(sidecars["tag_to_id.json"].read_text(encoding="utf-8"))
        self.category_map = json.loads(sidecars["tag_category.json"].read_text(encoding="utf-8"))
        self.id_to_tag = {int(v): k for k, v in self.tag_to_id.items()}
        print(f"Loaded {self.name} model from {model_path} with providers {self.model.get_providers()}")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image_size = int(self.config.get("image_size", 448))
        image = image.convert("RGB")
        w, h = image.size
        scale = image_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        canvas.paste(image, ((image_size - new_w) // 2, (image_size - new_h) // 2))
        arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return ((arr - mean) / std)[None, ...]

    def interrogate(self, image: Image.Image) -> Dict[str, List[Tuple[str, float]]]:
        if not hasattr(self, "model") or self.model is None:
            self.load()
        img_input = self.model.get_inputs()[0]
        output_name = self.model.get_outputs()[0].name
        logits = self.model.run([output_name], {img_input.name: self._preprocess(image)})[0][0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        result = {k: [] for k in ("rating", "general", "character", "copyright", "artist", "meta", "quality", "model")}
        cat_names = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}
        for idx, score in enumerate(probs):
            tag = self.id_to_tag.get(idx)
            if tag is None:
                continue
            category = cat_names.get(int(self.category_map.get(tag, 0)), "general")
            result[category].append((tag, float(score)))
        return result
