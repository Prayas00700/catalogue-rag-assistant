"""Step 2: ask the catalogue a question.

Usage:  python rag.py "black leather sneakers with a rubber outsole"
        python rag.py --list-models      (shows the Claude model IDs your API key can use)
Retrieves the closest products by vector search, asks Claude to answer ONLY from them,
then checks every product ID Claude cites against what was actually retrieved."""
import json, os, re, sys
import numpy as np
import config
from embedder import Embedder

ID_PATTERN = re.compile(r"BS-\d{5}", re.IGNORECASE)
NO_MATCH = "No matching product in the catalogue"

SYSTEM_PROMPT = f"""You are a product assistant for a footwear exporter. Buyers ask for styles; you answer
using ONLY the products listed inside <products>.

Rules:
- Recommend only products from <products>. Never invent a product ID, material or measurement.
- Cite every product you mention by its ID in square brackets, for example [BS-01234].
- For each product, say in a few words why it matches the request.
- If no listed product fits, reply exactly: "{NO_MATCH}." and suggest the closest alternative, if any, with its ID.
- Prices, buyers and suppliers are not available to you. If asked about them, say so.
- Before answering, check each requirement in the question against each product; answer only with the final result.

Example
Question: Do you have a brown suede boot?
<products>
Product BS-00412. Category: Boots; Upper material: Suede; Colour: Brown; Outsole: Rubber
Product BS-00977. Category: Sneakers; Upper material: Suede; Colour: Brown
</products>
Answer: Yes: [BS-00412] is a brown suede boot with a rubber outsole. [BS-00977] is also brown suede, but it is a sneaker, not a boot."""


class Catalogue:
    def __init__(self, index_dir=config.INDEX_DIR):
        self.vectors = np.load(os.path.join(index_dir, "vectors.npy"))
        self.embedder = Embedder.load(index_dir)
        self.items = [json.loads(l) for l in open(os.path.join(index_dir, "products.jsonl"), encoding="utf-8")]
        self.ids = [it["fields"]["Product ID"].upper() for it in self.items]
        self.by_id = {pid: i for i, pid in enumerate(self.ids)}
        self.fty = {}
        for i, it in enumerate(self.items):
            ref = it["fields"].get("Factory reference")
            if ref:
                self.fty.setdefault(ref.lower(), []).append(i)

    def search(self, question, k=config.TOP_K):
        """Vector search, with exact product-ID or factory-reference matches placed first."""
        exact = [self.by_id[m.upper()] for m in ID_PATTERN.findall(question) if m.upper() in self.by_id]
        for token in re.findall(r"[\w\-/]+", question.lower()):
            exact += self.fty.get(token, [])
        scores = self.vectors @ self.embedder.encode_query(question)
        ranked = [int(i) for i in np.argsort(-scores)]
        order = list(dict.fromkeys(exact + ranked))[:k]
        return [{"id": self.ids[i], "doc": self.items[i]["doc"], "fields": self.items[i]["fields"],
                 "score": float(scores[i]), "exact": i in exact} for i in order]


def build_prompt(question, hits):
    listing = "\n".join(h["doc"] for h in hits)
    return f"Question: {question}\n<products>\n{listing}\n</products>"


def ask_claude(question, hits):
    """Returns (answer_text, stop_reason)."""
    mock = os.environ.get("RAG_MOCK_LLM")
    if mock == "1":                              # offline test mode: no API call
        return "Matching products: " + ", ".join(f"[{h['id']}]" for h in hits[:2]) + ".", "end_turn"
    if mock == "empty":                          # offline test of an empty answer
        return "", "max_tokens"
    import anthropic
    client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY from the environment
    resp = client.messages.create(
        model=config.CLAUDE_MODEL, max_tokens=config.MAX_TOKENS, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    # Newer models may think before answering; keep only the text blocks
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    return text, resp.stop_reason


def verify(answer, hits, stop_reason="end_turn"):
    """Grounding check: every cited ID must be one of the retrieved products."""
    if not answer.strip():
        if stop_reason == "max_tokens":
            return "UNVERIFIED", "Claude used its whole token budget before writing an answer; raise MAX_TOKENS in config.py", []
        if stop_reason == "refusal":
            return "UNVERIFIED", "Claude declined to answer this request", []
        return "UNVERIFIED", f"Claude returned no answer text (stop reason: {stop_reason})", []
    if stop_reason == "max_tokens":
        return "UNVERIFIED", "the answer was cut off at the token limit; raise MAX_TOKENS in config.py", \
            list(dict.fromkeys(m.upper() for m in ID_PATTERN.findall(answer)))
    cited = list(dict.fromkeys(m.upper() for m in ID_PATTERN.findall(answer)))
    retrieved = {h["id"] for h in hits}
    outside = [c for c in cited if c not in retrieved]
    if outside:
        return "UNVERIFIED", f"cites products that were not retrieved: {', '.join(outside)}", cited
    if not cited:
        if NO_MATCH.lower() in answer.lower():
            return "NO MATCH", "Claude found no product that fits", cited
        return "UNVERIFIED", "names no product ID, so the answer can't be traced", cited
    return "VERIFIED", "every cited product was retrieved from the catalogue", cited


def answer(question, catalogue, k=config.TOP_K):
    hits = catalogue.search(question, k)
    text, stop_reason = ask_claude(question, hits)
    status, reason, cited = verify(text, hits, stop_reason)
    return {"question": question, "answer": text, "status": status, "reason": reason,
            "cited": cited, "retrieved": hits, "stop_reason": stop_reason}


def list_models():
    import anthropic
    for m in anthropic.Anthropic().models.list():
        print(m.id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python rag.py "your question"   or   python rag.py --list-models')
    if sys.argv[1] == "--list-models":
        list_models()
        sys.exit()
    result = answer(" ".join(sys.argv[1:]), Catalogue())
    print("\n" + (result["answer"] or "(no answer text)") + "\n")
    print(f"[{result['status']}] {result['reason']}")
    if result["stop_reason"] != "end_turn":
        print(f"Stop reason: {result['stop_reason']}")
    print("Retrieved:", ", ".join(h["id"] for h in result["retrieved"]))
