"""Step 4a: create test questions with known answers, straight from the catalogue.

Usage:  python make_testset.py [how_many]   (default 50) -> testset.csv
Each question describes one real product by three or four of its attributes. Every product
that shares exactly those attributes counts as a correct answer. Add your own rows to the CSV too."""
import csv, json, os, random, sys
import config

ATTRS = ["Colour", "Upper material", "Sub category", "Category", "Outsole", "Construction", "Toe shape", "Heel height"]
TEMPLATES = [
    "Do you have a {Colour} {Upper material} {Sub category} with a {Outsole} outsole?",
    "Looking for {Colour} {Sub category} in {Upper material}, {Construction} construction",
    "{Upper material} {Category} in {Colour} with a {Toe shape} toe",
    "Show me {Colour} {Upper material} styles with a {Outsole} outsole",
    "{Sub category} with {Heel height} heel height in {Colour}",
]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    items = [json.loads(l)["fields"] for l in open(os.path.join(config.INDEX_DIR, "products.jsonl"), encoding="utf-8")]
    rng = random.Random(42)
    rows, tries = [], 0
    while len(rows) < n and tries < n * 200:
        tries += 1
        product, template = rng.choice(items), rng.choice(TEMPLATES)
        needed = [a for a in ATTRS if "{" + a + "}" in template]
        if not all(product.get(a) for a in needed):
            continue
        question = template.format(**{a: product[a].lower() for a in needed})
        key = tuple(product[a].lower() for a in needed)
        expected = [p["Product ID"].upper() for p in items if tuple((p.get(a) or "").lower() for a in needed) == key]
        if len(expected) > 10:          # too generic to test retrieval meaningfully
            continue
        if any(r["question"] == question for r in rows):
            continue
        rows.append({"question": question, "expected_ids": ";".join(expected)})
    with open("testset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "expected_ids"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} test questions to testset.csv")


if __name__ == "__main__":
    main()
