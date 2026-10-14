"""Step 4b: measure the assistant.

Usage:  python evaluate.py                 retrieval only (free, no API calls)
        python evaluate.py --with-claude   also asks Claude and checks every answer
Reports: hit rate at 1 and at k (is a correct product retrieved?), and with --with-claude,
the share of answers that are VERIFIED and the share that cite a correct product."""
import csv, sys
import config
from rag import Catalogue, answer


def main():
    with_claude = "--with-claude" in sys.argv
    cat = Catalogue()
    rows = list(csv.DictReader(open("testset.csv", encoding="utf-8")))
    if not rows:
        sys.exit("testset.csv is empty. Run: python make_testset.py")
    out, hit1, hitk, verified, correct = [], 0, 0, 0, 0
    for r in rows:
        expected = {e.strip().upper() for e in r["expected_ids"].split(";") if e.strip()}
        if with_claude:
            res = answer(r["question"], cat)
            hits, status = res["retrieved"], res["status"]
            cited_ok = bool(expected & set(res["cited"]))
        else:
            hits, status, cited_ok = cat.search(r["question"]), "", False
        ids = [h["id"] for h in hits]
        h1, hk = ids[0] in expected, bool(expected & set(ids))
        hit1 += h1; hitk += hk
        verified += status == "VERIFIED"; correct += cited_ok
        out.append({"question": r["question"], "expected": ";".join(sorted(expected)), "retrieved": ";".join(ids),
                    "hit@1": h1, f"hit@{config.TOP_K}": hk, "status": status, "cites_correct": cited_ok})
    n = len(rows)
    print(f"Questions: {n}")
    print(f"Hit rate @1: {hit1 / n:.0%}    Hit rate @{config.TOP_K}: {hitk / n:.0%}")
    if with_claude:
        print(f"Answers VERIFIED: {verified / n:.0%}    Answers citing a correct product: {correct / n:.0%}")
    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("Details saved to eval_results.csv")


if __name__ == "__main__":
    main()
