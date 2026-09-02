#!/usr/bin/env python3
from pathlib import Path
import re

PROJECT = Path("/home/jova3528/private/MT/project_ukr_en")

RAW_TEST = PROJECT / "data/raw/test/Neulab-tedtalks_test-1-eng-ukr.ukr"
OUT = PROJECT / "results/bpe_segmentation_items.tsv"

BPE_TESTS = {
    "bpe5000": PROJECT / "data/bpe5000/bpe-data/test.uk",
    "bpe10000": PROJECT / "data/bpe10000/bpe-data/test.uk",
    "bpe30000": PROJECT / "data/bpe30000/bpe-data/test.uk",
}

ACRONYMS = [
    "США", "ООН", "ЄС", "НАТО", "ВІЛ", "СНІД", "ВВП", "ДНК", "РНК",
    "НАСА", "МРТ", "ФБР", "ШІ", "ЗМІ", "ЮНЕСКО", "ЮНІСЕФ", "ПТСР"
]

PLACE_PATTERNS = [
    r"Україн\w*",
    r"Київ\w*",
    r"Харків\w*",
    r"Львів\w*",
    r"Одес\w*",
    r"Крим\w*",
    r"Донбас\w*",
    r"Європ\w*",
    r"Америк\w*",
    r"Нью-Йорк\w*",
    r"Швеці\w*",
    r"Кита\w*",
    r"Інді\w*",
    r"Рим\w*",
]

def read_lines(path, expected_len=None):
    if not path.exists():
        if expected_len is None:
            return []
        return [""] * expected_len
    lines = path.read_text(encoding="utf-8").splitlines()
    if expected_len is not None and len(lines) != expected_len:
        raise ValueError(f"{path} has {len(lines)} lines, expected {expected_len}")
    return lines

def clean_tsv(text):
    return text.replace("\t", " ").replace("\n", " ").strip()

def bpe_words(bpe_line):
    words = []
    surface_parts = []
    bpe_parts = []

    for token in bpe_line.split():
        bpe_parts.append(token)
        if token.endswith("@@"):
            surface_parts.append(token[:-2])
        else:
            surface_parts.append(token)
            words.append(("".join(surface_parts), " ".join(bpe_parts)))
            surface_parts = []
            bpe_parts = []

    if surface_parts:
        words.append(("".join(surface_parts), " ".join(bpe_parts)))

    return words

def find_bpe_segment(item, bpe_line):
    if not bpe_line:
        return ""
    for surface, segmented in bpe_words(bpe_line):
        if surface == item:
            return segmented
    for surface, segmented in bpe_words(bpe_line):
        if item in surface or surface in item:
            return segmented
    return "NOT_FOUND"

def segmentation_status(item, segmented):
    if not segmented or segmented == "NOT_FOUND":
        return "not_found"
    if segmented == item:
        return "whole"
    if "@@" in segmented or " " in segmented:
        return "split"
    return "changed"

raw_lines = read_lines(RAW_TEST)
bpe_lines = {name: read_lines(path, len(raw_lines)) for name, path in BPE_TESTS.items()}

acronym_re = re.compile(
    r"(?<![А-Яа-яІіЇїЄєҐґ])("
    + "|".join(map(re.escape, ACRONYMS))
    + r")(?![А-Яа-яІіЇїЄєҐґ])"
)
place_re = re.compile(
    r"(?<![А-Яа-яІіЇїЄєҐґ])("
    + "|".join(PLACE_PATTERNS)
    + r")(?![А-Яа-яІіЇїЄєҐґ])"
)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    f.write(
        "line\ttype\titem\t"
        "bpe5000_status\tbpe5000_segmented\t"
        "bpe10000_status\tbpe10000_segmented\t"
        "bpe30000_status\tbpe30000_segmented\t"
        "source\n"
    )

    rows = 0
    for line_no, source in enumerate(raw_lines, 1):
        matches = []
        matches.extend(("acronym", item) for item in acronym_re.findall(source))
        matches.extend(("place", item) for item in place_re.findall(source))

        seen = set()
        for item_type, item in matches:
            key = (item_type, item)
            if key in seen:
                continue
            seen.add(key)

            row = [str(line_no), item_type, item]
            for bpe_name in ["bpe5000", "bpe10000", "bpe30000"]:
                segmented = find_bpe_segment(item, bpe_lines[bpe_name][line_no - 1])
                row.extend([segmentation_status(item, segmented), segmented])
            row.append(clean_tsv(source))
            f.write("\t".join(row) + "\n")
            rows += 1

print(f"Wrote {rows} rows to {OUT}")
