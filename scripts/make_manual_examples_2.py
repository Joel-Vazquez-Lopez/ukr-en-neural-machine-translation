import csv
from pathlib import Path

# Shared selected examples for manual analysis
examples = [
    (18, "proper_noun", "Нью-Йорк / foursquare"),
    (30, "proper_noun", "Деб Рой"),
    (37, "acronym", "MIT"),
    (39, "mixed", "Сполучені Штати / CNN / will.i.am / Андерсон Купер"),
    (53, "proper_noun", "Велика Британія"),
    (56, "proper_noun", "Оріндж / Каліфорнія"),
    (58, "proper_noun", "Сандра Дей О’Коннор"),
    (82, "organization", "Волл-Стрит Джорнел"),
    (114, "organization", "Товариство жінок-інженерів"),
    (135, "acronym", "MBA"),
    (265, "proper_noun", "Нізам блок / Аллама Ікбал Таун"),
    (286, "proper_noun", "Crash / Росія"),
    (322, "mixed", "Бьорн Сундін / Швеція / Інтерпол / Шалішкумар Джейн / США"),
    (326, "acronym", "США / Джейн"),
    (329, "acronym", "SQL"),
    (348, "acronym", "XOR"),
    (422, "mixed", "BNP Paribas / Китай / Сполучені Штати"),
    (603, "acronym", "ЄС / Європа"),
    (607, "acronym", "OECD"),
    (609, "acronym", "ЄС"),
    (963, "acronym", "TED"),
    (1018, "acronym", "TED"),
    (1101, "abbreviation", "ВІЛ / СНІД"),
    (1124, "acronym", "ООН"),
    (1138, "acronym", "ООН"),
    (1194, "acronym", "ООН"),
    (1214, "acronym", "США / Європа"),
    (1222, "abbreviation", "ВІЛ / СНІД"),
    (1223, "abbreviation", "ВІЛ / СНІД"),
    (1224, "abbreviation", "ВІЛ / СНІД"),
    (1395, "mixed", "Аргентина / Чилі / Болівія / ПАР / США / Японія"),
    (1601, "acronym", "NNT"),
    (1770, "acronym", "ООН"),
    (1771, "acronym", "ООН"),
    (2041, "acronym", "Вінт Серф / ARPA / NASA"),
    (2120, "mixed", "Уганда / Нью-Гемпшир / США / Бреттон-Вудська система / Світовий банк"),
    (2127, "mixed", "США / Європа / Світовий банк"),
    (2148, "mixed", "Філіппіни / Checkmyschool.org / Уганда / Біхар"),
    (2381, "abbreviation", "МРТ"),
    (2483, "proper_noun", "Вільям Свенсон / Почесна Медаль Конгресу"),
    (2546, "mixed", "Чарлі Кім / Next Jump / Нью-Йорк"),
    (2621, "acronym", "KFC"),
    (2703, "acronym", "США / Швеція"),
    (2731, "mixed", "Майкл Портер / Гарвардська школа бізнесу / Каролінський інститут / BCG / МОВРЗОЗ"),
    (2771, "mixed", "Майк Тайсон / TED"),
    (2793, "mixed", "GoTo.com / TED"),
    (2918, "mixed", "США / Центр Картера"),
    (2956, "organization", "Державний департамент США / Конгрес"),
    (3149, "mixed", "Уфіці / Нью-Йоркський музей сучасного мистецтва / MoMA / Ермітаж / Rijks / Музей ван Гога"),
    (3716, "organization", "Національна Організація з Океану та Атмосфери США"),
    (3724, "proper_noun", "Нью-Йорк / Тайм-Сквер / Хай-Лайн"),
]


def read_lines(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def read_hypotheses(path: Path):
    """
    Handles either:
    1. plain test.hyp with one translation per line
    2. raw fairseq-generate output containing H-<idx> lines
    """
    lines = read_lines(path)

    h_lines = [line for line in lines if line.startswith("H-")]
    if not h_lines:
        return lines

    parsed = []
    for line in h_lines:
        # fairseq H-line format:
        # H-123    -0.45    translated sentence
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        idx = int(parts[0].replace("H-", ""))
        hyp = parts[2]
        parsed.append((idx, hyp))

    parsed = sorted(parsed, key=lambda x: x[0])
    return [hyp for _, hyp in parsed]


# Change this if your final/latest BPE folder has a different name.
BPE_RUN_NAME = "bpe30000_clean_40ep_dropout0.3_seed1004"

base = Path("/home/jova3528/private/MT/project_ukr_en")

ukr_path = base / "data/raw/test/Neulab-tedtalks_test-1-eng-ukr.ukr"
ref_path = base / "data/raw/test/Neulab-tedtalks_test-1-eng-ukr.eng"
hyp_path = base / f"results/{BPE_RUN_NAME}/generation_best/test.sorted.hyp"

output_file = base / f"results/{BPE_RUN_NAME}/generation_best/manual_examples_{BPE_RUN_NAME}.csv"

ukr_lines = read_lines(ukr_path)
ref_lines = read_lines(ref_path)
hyp_lines = read_hypotheses(hyp_path)

print(f"Ukrainian source lines: {len(ukr_lines)}")
print(f"English reference lines: {len(ref_lines)}")
print(f"Hypothesis lines: {len(hyp_lines)}")
print(f"Using hypotheses from: {hyp_path}")

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "line",
        "category",
        "target_item",
        "ukr_source",
        "eng_reference",
        f"{BPE_RUN_NAME}_output",
        "annotator_1_label",
        "annotator_2_label",
        "comment",
    ])

    for line_number, category, target_item in examples:
        # Pelle raw test files have a first language-id/header line,
        # so idx = line_number aligns with the selected examples.
        # If your files do NOT have that first extra line, use idx = line_number - 1 instead.
        idx = line_number

        writer.writerow([
            line_number,
            category,
            target_item,
            ukr_lines[idx] if idx < len(ukr_lines) else "",
            ref_lines[idx] if idx < len(ref_lines) else "",
            hyp_lines[idx] if idx < len(hyp_lines) else "",
            "",
            "",
            "",
        ])

print(f"Done. Wrote {len(examples)} examples to {output_file}")