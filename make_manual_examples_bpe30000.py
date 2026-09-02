import csv
from pathlib import Path

# Shared selected examples for manual analysis
# Format: line_number, category, target_item
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

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

base = Path("/home/jova3528/private/MT/project_ukr_en")

ukr_lines = read_lines(base / "data/raw/test/Neulab-tedtalks_test-1-eng-ukr.ukr")
ref_lines = read_lines(base / "data/raw/test/Neulab-tedtalks_test-1-eng-ukr.eng")
bpe_lines = read_lines(base / "results/bpe30000/translations/test.hyp")

output_file = base / "results/bpe30000/translations/manual_examples_bpe30000.csv"

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "line",
        "category",
        "target_item",
        "ukr_source",
        "eng_reference",
        "bpe30000_output",
        "annotator_1_label",
        "annotator_2_label",
        "comment",
    ])

    for line_number, category, target_item in examples:
        # Pelle raw test files have a first language-id line, so this offset aligns with the selected examples.
        idx = line_number

        writer.writerow([
            line_number,
            category,
            target_item,
            ukr_lines[idx] if idx < len(ukr_lines) else "",
            ref_lines[idx] if idx < len(ref_lines) else "",
            bpe_lines[idx] if idx < len(bpe_lines) else "",
            "",
            "",
            "",
        ])

print(f"Done. Wrote {len(examples)} examples to {output_file}")
