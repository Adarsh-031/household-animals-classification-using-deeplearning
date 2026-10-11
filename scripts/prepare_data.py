import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "train"
OUTPUT = ROOT / "data" / "splits"
SEED = 42
LABELS = {"cat": 0, "dog": 1}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not SOURCE.is_dir():
        raise SystemExit(f"Missing dataset directory: {SOURCE}")

    files = sorted(
        path for path in SOURCE.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise SystemExit(f"No images found in {SOURCE}")

    groups = defaultdict(list)
    excluded = []

    print(f"Checking {len(files):,} images...", flush=True)

    for index, path in enumerate(files, start=1):
        relative = path.relative_to(ROOT).as_posix()
        match = re.fullmatch(
            r"(cat|dog)\.\d+\.(jpg|jpeg|png)", path.name, re.IGNORECASE
        )

        if not match:
            excluded.append({
                "path": relative,
                "reason": "Unexpected filename; label cannot be determined",
            })
            continue

        animal = match.group(1).lower()

        try:
            # Verify the file, then reopen and fully decode its pixels.
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb.load()
                width, height = rgb.size

                # Compare decoded RGB pixels, including image dimensions.
                digest = hashlib.sha256()
                digest.update(f"{width}x{height}:".encode("ascii"))
                digest.update(rgb.tobytes())
                fingerprint = digest.hexdigest()

            groups[fingerprint].append({
                "path": relative,
                "class": animal,
                "label": LABELS[animal],
                "width": width,
                "height": height,
                "sha256_rgb": fingerprint,
            })

        except Exception as error:
            excluded.append({
                "path": relative,
                "reason": f"Unreadable image: {type(error).__name__}: {error}",
            })

        if index % 1000 == 0:
            print(f"Checked {index:,}/{len(files):,}", flush=True)

    retained = []

    for entries in groups.values():
        if len({entry["label"] for entry in entries}) > 1:
            # Identical pixels with contradictory labels require review.
            for entry in entries:
                excluded.append({
                    "path": entry["path"],
                    "reason": "Identical image has conflicting cat/dog labels",
                })
            continue

        retained.append(entries[0])

        for duplicate in entries[1:]:
            excluded.append({
                "path": duplicate["path"],
                "reason": f"Exact duplicate of {entries[0]['path']}",
            })

    splits = {"train": [], "val": [], "test": []}

    for animal, label in LABELS.items():
        rows = sorted(
            (row for row in retained if row["label"] == label),
            key=lambda row: row["path"],
        )

        if len(rows) < 25:
            raise SystemExit(f"Too few valid images for {animal}: {len(rows)}")

        random.Random(SEED + label).shuffle(rows)

        train_count = int(len(rows) * 0.80)
        val_count = int(len(rows) * 0.16)

        splits["train"].extend(rows[:train_count])
        splits["val"].extend(rows[train_count:train_count + val_count])
        splits["test"].extend(rows[train_count + val_count:])

    # Explicitly check that no path or exact image crosses splits.
    for field in ("path", "sha256_rgb"):
        seen = set()
        for name, rows in splits.items():
            values = {row[field] for row in rows}
            if len(values) != len(rows) or seen.intersection(values):
                raise RuntimeError(f"Duplicate {field} found in split {name}")
            seen.update(values)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = ["path", "class", "label", "width", "height", "sha256_rgb"]

    counts = {}
    for name, rows in splits.items():
        rows.sort(key=lambda row: row["path"])
        write_csv(OUTPUT / f"{name}.csv", rows, fields)
        counts[name] = dict(Counter(row["class"] for row in rows))

    write_csv(
        OUTPUT / "excluded.csv", excluded, ["path", "reason"]
    )

    summary = {
        "seed": SEED,
        "label_mapping": LABELS,
        "candidate_images": len(files),
        "retained_images": len(retained),
        "excluded_images": len(excluded),
        "split_counts": counts,
        "duplicate_check": "Exact decoded RGB pixels; no near-duplicate check",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nCompleted!")
    for name, rows in splits.items():
        print(f"{name:5}: {len(rows):6,} images | {counts[name]}")
    print(f"Excluded: {len(excluded):,} — see data/splits/excluded.csv")
    print("Labels: cat=0, dog=1")
    print("Original images and test1 were left untouched.")


if __name__ == "__main__":
    main()