import csv
from pathlib import Path
from typing import List, Tuple


def load_selected_tags(path: Path) -> Tuple[List[str], List[int]]:
    """Load tag names/categories while preserving row order exactly."""
    names: List[str] = []
    categories: List[int] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])

        if "name" not in fieldnames:
            raise ValueError(
                f"Tag metadata file {path} does not contain a 'name' column"
            )

        for row_number, row in enumerate(reader, start=2):
            name = row.get("name")
            if name is None or not name.strip():
                raise ValueError(
                    f"Tag metadata file {path} contains an empty tag name "
                    f"at CSV row {row_number}"
                )

            raw_category = row.get("category", "0")
            try:
                category = int(raw_category or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid tag category {raw_category!r} "
                    f"at CSV row {row_number} in {path}"
                ) from exc

            names.append(name)
            categories.append(category)

    return names, categories
