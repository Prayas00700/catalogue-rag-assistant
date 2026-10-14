"""Step 1: build the search index from the catalogue export.

Usage:  python ingest.py path/to/All_prods.xlsx
Reads the 49-column catalogue, drops prices, buyer and supplier details, writes one text
document per product, embeds them locally and saves everything in ./index/."""
import json, os, sys
import numpy as np
import pandas as pd
import config
from embedder import Embedder


def clean(value):
    """Return the value as text, or None if it is one of the catalogue's 'empty' markers."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in config.NULL_TOKENS:
        return None
    if text.endswith(".0") and text[:-2].isdigit():   # Excel turns 25 into 25.0
        text = text[:-2]
    return text


def read_catalogue(path):
    xl = pd.ExcelFile(path)
    sheet = "Data Sheet" if "Data Sheet" in xl.sheet_names else xl.sheet_names[0]
    df = xl.parse(sheet, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    if "Product ID" not in df.columns:
        sys.exit(f"No 'Product ID' column found in sheet '{sheet}'. Columns: {list(df.columns)[:10]}...")
    dropped = [c for c in config.EXCLUDED_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    return df, sheet, dropped


def product_record(row):
    fields = {}
    for col, label in config.INDEXED_FIELDS:
        value = clean(row.get(col))
        if value is None and col in config.FALLBACK_COLUMNS:
            value = clean(row.get(config.FALLBACK_COLUMNS[col]))
        if value is not None:
            fields[label] = value
    return fields


def to_document(fields):
    """One readable line per product: what gets embedded and what Claude sees."""
    pid = fields.get("Product ID", "?")
    rest = "; ".join(f"{k}: {v}" for k, v in fields.items() if k != "Product ID")
    return f"Product {pid}. {rest}"


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python ingest.py path/to/catalogue.xlsx")
    df, sheet, dropped = read_catalogue(sys.argv[1])
    records, docs = [], []
    for _, row in df.iterrows():
        fields = product_record(row)
        if "Product ID" not in fields:
            continue
        records.append(fields)
        docs.append(to_document(fields))
    if not docs:
        sys.exit("No products found.")
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    emb = Embedder()
    vectors = emb.fit_encode(docs)
    np.save(os.path.join(config.INDEX_DIR, "vectors.npy"), vectors)
    emb.save(config.INDEX_DIR)
    with open(os.path.join(config.INDEX_DIR, "products.jsonl"), "w", encoding="utf-8") as f:
        for rec, doc in zip(records, docs):
            f.write(json.dumps({"fields": rec, "doc": doc}, ensure_ascii=False) + "\n")
    meta = {"source": os.path.basename(sys.argv[1]), "sheet": sheet, "products": len(docs),
            "embedder": emb.kind, "embed_model": config.EMBED_MODEL if emb.kind == "st" else "tfidf",
            "dimensions": int(vectors.shape[1]), "excluded_columns_found": dropped}
    with open(os.path.join(config.INDEX_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Indexed {len(docs)} products from sheet '{sheet}' ({vectors.shape[1]} dimensions).")
    print(f"Kept out of the index: {', '.join(dropped) if dropped else 'none of the excluded columns were present'}")


if __name__ == "__main__":
    main()
