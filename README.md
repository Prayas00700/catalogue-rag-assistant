# Catalogue Assistant: a RAG project over the product catalogue

Ask for footwear styles in plain English. The assistant:

1. finds the closest products by **vector search** over local embeddings (BAAI/bge-small-en-v1.5),
   with exact product-ID and factory-reference matches placed first;
2. sends only those products to **Claude**, with a system prompt that allows answers only from them,
   requires every product to be cited by ID, and includes one worked example;
3. runs a **grounding check**: any answer that cites a product it was not given is marked UNVERIFIED;
4. never indexes prices, buyer details or supplier codes, so it can never reveal them;
5. comes with a **test set and evaluator** that measure retrieval hit rate and answer quality.

## Files
| File | Purpose |
| --- | --- |
| config.py | All settings: models, columns kept, columns excluded |
| ingest.py | Builds the index from the catalogue export |
| embedder.py | Local embeddings (and a TF-IDF fallback for offline testing) |
| rag.py | Search, the Claude prompt, and the grounding check; also a command-line tool |
| app.py | The web app (Streamlit) |
| make_testset.py | Generates test questions with known answers from the catalogue |
| evaluate.py | Measures hit rate and, optionally, Claude's answers |

## Before you start
- Use your 49-column catalogue export (All_prods.xlsx, or BasantWiz's Excel export in the 49-column format).
- Ask the business before sending product specifications to an external AI service. Only the ~8 products
  retrieved for each question are sent to Claude, and prices, buyers and suppliers never are.
- You need an Anthropic API key with credit (console.anthropic.com).

## Steps (Mac Terminal, one command at a time; see the chat for each step explained)
1. `cd ~/catalogue_rag`
2. `python3 -m venv .venv`
3. `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. `export ANTHROPIC_API_KEY="your-key"`
6. `python rag.py --list-models`  then, if needed, `export CLAUDE_MODEL="model-id-from-the-list"`
7. `python ingest.py ~/path/to/All_prods.xlsx`   (check it says 7,191 products)
8. `python rag.py "black leather sneakers with a rubber outsole"`
9. `streamlit run app.py`
10. `python make_testset.py 50`
11. `python evaluate.py`
12. `python evaluate.py --with-claude`

## Offline test (no model download, no API key)
`RAG_EMBEDDER=tfidf RAG_MOCK_LLM=1 python ingest.py fake_catalogue.xlsx` and the same variables for the other scripts.
