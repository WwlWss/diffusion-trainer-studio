import tempfile
import unittest
from pathlib import Path

from mikazuki.tagger.interrogators.tag_metadata import load_selected_tags


ROOT = Path(__file__).resolve().parents[1]


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

    def test_selected_tags_reject_empty_names_instead_of_shifting_indices(self):
        path = self._write_csv(
            "name,category\n"
            "1girl,0\n"
            ",4\n"
        )

        with self.assertRaisesRegex(ValueError, "empty tag name"):
            load_selected_tags(path)

    def test_selected_tags_reject_invalid_categories(self):
        path = self._write_csv(
            "name,category\n"
            "1girl,not-a-number\n"
        )

        with self.assertRaisesRegex(ValueError, "Invalid tag category"):
            load_selected_tags(path)

    def test_selected_tags_require_name_column(self):
        path = self._write_csv(
            "tag,category\n"
            "1girl,0\n"
        )

        with self.assertRaisesRegex(ValueError, "does not contain a 'name' column"):
            load_selected_tags(path)

    def test_csv_backed_taggers_do_not_use_pandas_row_iteration(self):
        for filename in ("animetimm.py", "pixai.py"):
            source = (
                ROOT / "mikazuki/tagger/interrogators" / filename
            ).read_text(encoding="utf-8")
            self.assertNotIn("import pandas", source, filename)
            self.assertNotIn(".iterrows()", source, filename)
            self.assertIn("load_selected_tags", source, filename)
            self.assertIn("output/tag count mismatch", source, filename)
            self.assertIn("np.asarray(probs).reshape(-1)", source, filename)


if __name__ == "__main__":
    unittest.main()
