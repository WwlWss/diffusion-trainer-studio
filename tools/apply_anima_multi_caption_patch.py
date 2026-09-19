"""Apply optional Multi-Caption support to a staged pinned sd-scripts tree.

The real sd-scripts submodule stays pristine.  This tool patches only an
isolated runtime copy and fails closed whenever a reviewed source anchor drifts.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tools.apply_anima_qwen3_sd_scripts_patch import EXPECTED_SD_SCRIPTS_HEAD, replace_once


def patch_args(text: str) -> str:
    anchor = '''    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ"
    )
'''
    replacement = anchor + '''    parser.add_argument(
        "--multi_caption_config",
        type=str,
        default=None,
        help="optional DTS Multi-Caption JSON sidecar; absent keeps the legacy caption path unchanged",
    )
'''
    return replace_once(text, anchor, replacement, "Multi-Caption dataset CLI argument")


def patch_dataset(text: str) -> str:
    text = replace_once(
        text,
        '''        self.replacements = {}

        self.tokenize_strategy = None
''',
        '''        self.replacements = {}

        # Optional Multi-Caption is attached only after the historical Dataset
        # has been fully constructed. None means the exact Standard path.
        self.multi_caption_resolver = None

        self.tokenize_strategy = None
''',
        "Multi-Caption resolver state",
    )
    text = replace_once(
        text,
        '''    def is_text_encoder_output_cacheable(self, cache_supports_dropout: bool = False):
        return all(
''',
        '''    def is_text_encoder_output_cacheable(self, cache_supports_dropout: bool = False):
        if self.multi_caption_resolver is not None:
            return False
        return all(
''',
        "Multi-Caption TE cache guard",
    )
    text = replace_once(
        text,
        '''        )

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
''',
        '''        )

    def set_multi_caption_resolver(self, resolver):
        self.multi_caption_resolver = resolver

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
''',
        "Multi-Caption resolver setter",
    )
    text = replace_once(
        text,
        '''            if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]  # remove batch dimension
''',
        '''            if tokenization_required:
                if self.multi_caption_resolver is None or image_info.is_reg:
                    caption = self.process_caption(subset, image_info.caption)
                else:
                    resolved_caption = self.multi_caption_resolver.choose(
                        image_path=image_info.absolute_path,
                        image_key=image_info.image_key,
                    )
                    caption = self.process_caption(
                        resolved_caption.processing,
                        resolved_caption.caption,
                    )
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]  # remove batch dimension
''',
        "Multi-Caption exposure resolver",
    )
    text = replace_once(
        text,
        '''    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    # def make_buckets(self):
''',
        '''    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    def set_multi_caption_resolver(self, resolver):
        for dataset in self.datasets:
            dataset.set_multi_caption_resolver(resolver)

    # def make_buckets(self):
''',
        "Multi-Caption DatasetGroup setter",
    )
    return text


def _dataset_attach_block(indent: str) -> str:
    return f'''{indent}if args.multi_caption_config:
{indent}    if args.dataset_class is not None:
{indent}        raise ValueError("Multi-Caption v1 does not support custom dataset_class")
{indent}    from library.multi_caption import configure_multi_caption_dataset_groups
{indent}    configure_multi_caption_dataset_groups(
{indent}        args.multi_caption_config,
{indent}        train_dataset_group,
{indent}        val_dataset_group,
{indent}    )

'''


def patch_train_network(text: str) -> str:
    anchor = '''            val_dataset_group = None  # placeholder until validation dataset supported for arbitrary

        current_epoch = Value("i", 0)
'''
    replacement = (
        '''            val_dataset_group = None  # placeholder until validation dataset supported for arbitrary

'''
        + _dataset_attach_block("        ")
        + '''        current_epoch = Value("i", 0)
'''
    )
    return replace_once(text, anchor, replacement, "Multi-Caption common network trainer attach")


def patch_anima_train(text: str) -> str:
    anchor = '''        val_dataset_group = None

    current_epoch = Value("i", 0)
'''
    replacement = (
        '''        val_dataset_group = None

'''
        + _dataset_attach_block("    ")
        + '''    current_epoch = Value("i", 0)
'''
    )
    return replace_once(text, anchor, replacement, "Multi-Caption Anima full trainer attach")


def patch_files(sd_scripts_dir: Path, multi_caption_source: Path | None = None) -> dict[Path, str]:
    patchers = {
        sd_scripts_dir / "library/args.py": patch_args,
        sd_scripts_dir / "library/dataset.py": patch_dataset,
        sd_scripts_dir / "train_network.py": patch_train_network,
        sd_scripts_dir / "anima_train.py": patch_anima_train,
    }
    result: dict[Path, str] = {}
    for path, patcher in patchers.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        original = path.read_text(encoding="utf-8-sig")
        result[path] = patcher(original)

    if multi_caption_source is None:
        multi_caption_source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "dev"
            / "library"
            / "multi_caption.py"
        )
    if not multi_caption_source.is_file():
        raise FileNotFoundError(multi_caption_source)
    result[sd_scripts_dir / "library/multi_caption.py"] = multi_caption_source.read_text(encoding="utf-8")
    return result


def validate_patch(sd_scripts_dir: Path, multi_caption_source: Path | None = None) -> None:
    for path, source in patch_files(sd_scripts_dir, multi_caption_source).items():
        ast.parse(source, filename=str(path))
