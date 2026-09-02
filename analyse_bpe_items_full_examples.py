#!/usr/bin/env python3
from pathlib import Path
import re
from collections import Counter

PROJECT = Path("/home/jova3528/private/MT/project_ukr_en")

RAW_TEST = PROJECT / "data/raw/test/Neulab-tedtalks_test-1-eng-ukr.ukr"
ACRONYM_FILE = PROJECT / "results/acronyms_list_clean.txt"

OUT_DETAILS = PROJECT / "results/bpe_segmentation_full_examples.tsv"
OUT_SUMMARY = PROJECT / "results/bpe_segmentation_full_summary.tsv"

BPE_TESTS = {
    "bpe5000": PROJECT / "data/bpe5000/bpe-data/test.uk",
    "bpe10000": PROJECT / "data/bpe10000/bpe-data/test.uk",
    "bpe30000": PROJECT / "data/bpe30000/bpe-data/test.uk",
}

# These are the place/entity roots we are explicitly interested in.
# Add/remove items here if you want to narrow or expand the analysis.
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

UK_BOUNDARY_LEFT = r"(?<![А-Яа-яІіЇїЄєҐґ])"
UK_BOUNDARY_RIGHT = r"(?![А-Яа-яІіЇїЄєҐґ])"


def read_lines(path, expected_len=None):
    if not path.exists():
        raise FileNotFoundError(path)
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
        return "NOT_FOUND"

    words = bpe_words(bpe_line)

    # Exact reconstructed surface match.
    for surface, segmented in words:
        if surface == item:
            return segmented

    # Fallback for small punctuation / inflection mismatches.
    for surface, segmented in words:
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


def load_acronyms():
    if not ACRONYM_FILE.exists():
        raise FileNotFoundError(
            f"{ACRONYM_FILE} does not exist. Create it first with the grep command."
        )
    acronyms = [
        line.strip()
        for line in ACRONYM_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return acronyms


def main():
    raw_lines = read_lines(RAW_TEST)
    bpe_lines = {name: read_lines(path, len(raw_lines)) for name, path in BPE_TESTS.items()}

    acronyms = load_acronyms()

    acronym_re = re.compile(
        UK_BOUNDARY_LEFT
        + r"("
        + "|".join(map(re.escape, sorted(acronyms, key=len, reverse=True)))
        + r")"
        + UK_BOUNDARY_RIGHT
    )

    place_re = re.compile(
        UK_BOUNDARY_LEFT
        + r"("
        + "|".join(PLACE_PATTERNS)
        + r")"
        + UK_BOUNDARY_RIGHT
    )

    counts = {
        "bpe5000": Counter(),
        "bpe10000": Counter(),
        "bpe30000": Counter(),
    }

    type_counts = Counter()

    OUT_DETAILS.parent.mkdir(parents=True, exist_ok=True)

    with OUT_DETAILS.open("w", encoding="utf-8") as f:
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

            for item in acronym_re.findall(source):
                matches.append(("acronym", item))

            for item in place_re.findall(source):
                matches.append(("place", item))

            # Avoid duplicate rows for the same item repeated in the same sentence.
            seen = set()

            for item_type, item in matches:
                key = (item_type, item)
                if key in seen:
                    continue
                seen.add(key)

                row = [str(line_no), item_type, item]
                type_counts[item_type] += 1

                for bpe_name in ["bpe5000", "bpe10000", "bpe30000"]:
                    segmented = find_bpe_segment(item, bpe_lines[bpe_name][line_no - 1])
                    status = segmentation_status(item, segmented)
                    counts[bpe_name][status] += 1
                    row.extend([status, segmented])

                row.append(clean_tsv(source))
                f.write("\t".join(row) + "\n")
                rows += 1

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        f.write("bpe_size\twhole\tsplit\tchanged\tnot_found\ttotal_found_or_changed\twhole_percentage\n")

        for bpe_name in ["bpe5000", "bpe10000", "bpe30000"]:
            c = counts[bpe_name]
            whole = c["whole"]
            split = c["split"]
            changed = c["changed"]
            not_found = c["not_found"]
            denom = whole + split + changed
            pct = 100 * whole / denom if denom else 0.0

            f.write(
                f"{bpe_name}\t{whole}\t{split}\t{changed}\t{not_found}\t{denom}\t{pct:.1f}%\n"
            )

    print(f"Raw test lines: {len(raw_lines)}")
    print(f"Acronyms loaded: {len(acronyms)}")
    print(f"Rows written: {rows}")
    print(f"Type counts: {dict(type_counts)}")
    print(f"Details written to: {OUT_DETAILS}")
    print(f"Summary written to: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
