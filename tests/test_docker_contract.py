from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DockerContractTests(unittest.TestCase):
    def _read(self, name: str) -> str:
        return (ROOT / name).read_text(encoding="utf-8")

    def test_default_dockerfile_builds_current_checkout(self):
        content = self._read("Dockerfile")
        self.assertIn("COPY . .", content)
        self.assertNotIn("git clone", content)
        self.assertIn("frontend/dist/index.html", content)
        self.assertIn("mikazuki/dataset-tag-editor/scripts/launch.py", content)
        self.assertIn("sd-scripts/anima_train_network.py", content)
        self.assertIn("pip install -r requirements.txt", content)
        self.assertNotIn("scripts/stable", content)
        self.assertNotIn("scripts/dev", content)

    def test_china_dockerfile_keeps_same_checkout_and_dependency_contract(self):
        content = self._read("Dockerfile-for-Mainland-China")
        self.assertIn("COPY . .", content)
        self.assertNotIn("git clone", content)
        self.assertIn("pip install -r requirements.txt", content)
        self.assertNotIn("WORKDIR /app/diffusion-trainer-studio/scripts/stable", content)
        self.assertNotIn("WORKDIR /app/diffusion-trainer-studio/scripts/dev", content)
        self.assertIn('python -c "from opencv_fixer import AutoFix; AutoFix()" && \\', content)
        self.assertIn("pip install opencv-python-headless && \\", content)
        self.assertIn("apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libgl1", content)

    def test_dockerignore_excludes_local_runtime_data_without_hiding_submodules(self):
        content = self._read(".dockerignore")
        self.assertIn(".git", content)
        self.assertIn("venv", content)
        self.assertIn("logs/*", content)
        self.assertIn("output/*", content)
        self.assertIn("sd-models/*", content)
        self.assertNotIn("frontend", content)
        self.assertNotIn("sd-scripts", content)
        self.assertNotIn("mikazuki/dataset-tag-editor", content)


if __name__ == "__main__":
    unittest.main()
