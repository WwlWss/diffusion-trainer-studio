import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

from mikazuki.tagger.interrogators.base import Interrogator


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
        # Import PyTorch before ONNX Runtime so ORT can reuse the CUDA/cuDNN DLLs
        # bundled with the project's PyTorch installation on Windows.
        import torch
        import onnxruntime as ort

        if not torch.cuda.is_available():
            raise RuntimeError(
                "AnimeTimm requires a usable CUDA GPU in DTS. PyTorch reports CUDA is unavailable; "
                "CPU tagging fallback is disabled."
            )

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as exc:
                raise RuntimeError(f"ONNX Runtime CUDA DLL preload failed: {exc}") from exc

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "AnimeTimm requires ONNX Runtime CUDAExecutionProvider, but it is not available. "
                f"Available providers: {available}. CPU tagging fallback is disabled."
            )

        model_path, tags_path, preprocess_path = self.download()
        session_options = ort.SessionOptions()
        # ORT normally assigns unsupported nodes to CPU even when only the CUDA
        # provider is requested. This config makes session creation fail instead,
        # so a nominally-GPU tagger can never perform hidden CPU graph fallback.
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        try:
            self.model = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=["CUDAExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                "AnimeTimm could not create a CUDA-only ONNX Runtime session. "
                "DTS does not fall back to CPU tagging; verify the ONNX Runtime/CUDA/cuDNN installation. "
                f"Original error: {exc}"
            ) from exc

        active = self.model.get_providers()
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                f"AnimeTimm CUDA session was not activated (active providers: {active}). "
                "CPU tagging fallback is disabled."
            )

        self.tags = pd.read_csv(tags_path)
        self.preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
        print(f"Loaded {self.name} model from {model_path} with providers {active}")

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
        for (_, row), score in zip(self.tags.iterrows(), probs):
            category = category_map.get(int(row.get("category", 0)), "general")
            result[category].append((str(row["name"]), float(score)))
        return result
