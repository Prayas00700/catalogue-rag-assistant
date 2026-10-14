"""Settings shared by every script. Change values here, not inside the scripts."""
import os

# Where the built index lives (created by ingest.py)
INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "index")

# Embedding model, run locally on your Mac (downloaded once, then cached)
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Claude model used to write answers. Confirm the exact ID in your Anthropic Console,
# or run:  python rag.py --list-models
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")

TOP_K = 8            # products retrieved per question
MAX_TOKENS = 4000    # room for Claude's thinking plus its answer (thinking tokens count toward this limit)

# Values the catalogue uses to mean "empty"
NULL_TOKENS = {"", "-", "n.a", "na", "n/a", "nil", "none", "0", "nan"}

# Columns that become searchable text, in this order: (column, label shown to Claude)
INDEXED_FIELDS = [
    ("Product ID", "Product ID"), ("FTY Reference", "Factory reference"), ("Art #", "Article no."),
    ("Category", "Category"), ("Sub Category", "Sub category"), ("Product Type", "Product type"),
    ("Season", "Season"), ("Brand", "Brand"), ("Construction", "Construction"),
    ("Toe Shape", "Toe shape"), ("Last", "Last"), ("Width", "Width"), ("Heel Height", "Heel height"),
    ("Size Range", "Size range"), ("Upper Material", "Upper material"), ("Color", "Colour"),
    ("Lining", "Lining"), ("Socks", "Sockliner"), ("Outsole", "Outsole"), ("Sole", "Sole material"),
    ("Group Name", "Group"), ("Description", "Description"),
]

# If the first column of a pair is empty, use the second (the template repeats some fields)
FALLBACK_COLUMNS = {
    "Sub Category": "Sub-Category", "Outsole": "Outsole Name", "Sole": "Material Of Outsole",
}

# Never indexed, so the assistant can never reveal them:
# prices and order terms, buyer details, supplier identity, platform internals
EXCLUDED_COLUMNS = [
    "Price from Vendor_Currency", "Price from Vendor", "Price from Vendor_Unit",
    "Selling Price_Currency", "Selling Price", "Selling Price_Unit", "MOQ", "Vendor Name",
    "Customer Code", "Cust Reference", "Exclusive For", "Adopted",
    "Supplier Code", "DBID", "Duplicate Check", "Search keywords", "Image URL", "Image",
]
