import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mikazuki.tagger.interrogators.animetimm import AnimeTimmInterrogator
from mikazuki.tagger.interrogators.pixai import PixAITaggerInterrogator
from mikazuki.tagger.interrogators.tag_metadata import load_selected_tags


class _TensorInfo:
    def __init__(self, name):
        self.name = name


class _AnimeModel:
    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype=np.float32)

    def get_inputs(self):
        return [_TensorInfo("image")]

    def run(self, output_names, inputs):
        return [self.logits]


class _PixAIModel:
    def __init__(self, probs):
        self.probs = np.asarray([probs], dtype=np.float32)

    def get_inputs(self):
        info = _TensorInfo("image")
        info.shape = [1, 3, 4, 4]
        return [info]

    def get_outputs(self):
        return [_TensorInfo("aux0"), _TensorInfo("aux1"), _TensorInfo("pred")]

    def run(self, output_names, inputs):
        return [self.probs]


class TagMetadataTests(unittest.TestCase):
    def _write_csv(self, text):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "selected_tags.csv"
        path.write_text(text, encoding="utf-8")
        self.addCleanup(tmp.cleanup)
        return path

    def test_selected_tags_preserve_row_order_and_default_category(self):
        path = self._write_csv(
            "name,category\n"
            "1girl,0\n"
            "hatsune_miku,4\n"
            "untagged,\n"
        )

        names, categories = load_selected_tags(path)

        self.assertEqual(names, ["1girl", "hatsune_miku", "untagged"])
        self.assertEqual(categories, [0, 4, 0])

    def test_selected_tags_reject_invalid_rows_instead_of_shifting_indices(self):
        path = self._write_csv(
            "name,category\n"
            "1girl,0\n"
            ",4\n"
        )

        with self.assertRaisesRegex(ValueError, "empty tag name"):
            load_selected_tags(path)

    def test_animetimm_maps_scores_without_pandas_iterrows(self):
        interrogator = AnimeTimmInterrogator("test", "unused")
        interrogator.model = _AnimeModel([0.0, 1.0, -1.0])
        interrogator.tag_names = ["general_tag", "character_tag", "quality_tag"]
        interrogator.tag_categories = [0, 4, 2]
        interrogator._preprocess = lambda image: np.zeros((1, 3, 4, 4), dtype=np.float32)

        result = interrogator.interrogate(Image.new("RGB", (4, 4)))

        self.assertEqual(result["general"][0][0], "general_tag")
        self.assertEqual(result["character"][0][0], "character_tag")
        self.assertEqual(result["quality"][0][0], "quality_tag")
        self.assertAlmostEqual(result["general"][0][1], 0.5, places=6)

    def test_animetimm_rejects_output_tag_count_mismatch(self):
        interrogator = AnimeTimmInterrogator("test", "unused")
        interrogator.model = _AnimeModel([0.0, 1.0])
        interrogator.tag_names = ["a", "b", "c"]
        interrogator.tag_categories = [0, 0, 0]
        interrogator._preprocess = lambda image: np.zeros((1, 3, 4, 4), dtype=np.float32)

        with self.assertRaisesRegex(RuntimeError, "output/tag count mismatch"):
            interrogator.interrogate(Image.new("RGB", (4, 4)))

    def test_pixai_maps_scores_without_pandas_iterrows(self):
        interrogator = PixAITaggerInterrogator("test", "unused")
        interrogator.model = _PixAIModel([0.2, 0.8])
        interrogator.tag_names = ["general_tag", "character_tag"]
        interrogator.tag_categories = [0, 4]
        interrogator._preprocess = lambda image: np.zeros((1, 3, 4, 4), dtype=np.float32)

        result = interrogator.interrogate(Image.new("RGB", (4, 4)))

        self.assertEqual(result["general"], [("general_tag", 0.20000000298023224)])
        self.assertEqual(result["character"], [("character_tag", 0.800000011920929)])


if __name__ == "__main__":
    unittest.main()
