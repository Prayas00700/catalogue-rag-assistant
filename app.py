"""Step 3: the web app.  Run:  streamlit run app.py"""
import json, os
import pandas as pd
import streamlit as st
import config
from rag import Catalogue, answer

st.set_page_config(page_title="Catalogue Assistant", layout="wide")
st.title("Catalogue Assistant")
st.caption("Ask for styles in plain English. Answers come only from the catalogue, and every cited product is checked.")


@st.cache_resource
def load_catalogue():
    return Catalogue()


if not os.path.exists(os.path.join(config.INDEX_DIR, "vectors.npy")):
    st.error("No index found. Run:  python ingest.py path/to/catalogue.xlsx")
    st.stop()

cat = load_catalogue()
meta = json.load(open(os.path.join(config.INDEX_DIR, "meta.json")))
st.sidebar.write(f"**Products indexed:** {meta['products']:,}")
st.sidebar.write(f"**Embedding model:** {meta['embed_model']}")
st.sidebar.write(f"**Claude model:** {config.CLAUDE_MODEL}")
k = st.sidebar.slider("Products retrieved per question", 3, 15, config.TOP_K)
st.sidebar.write("**Never indexed:** prices, buyer details, supplier codes.")

question = st.text_input("Question", placeholder="e.g. black leather sneakers with a rubber outsole")
if question:
    with st.spinner("Searching and asking Claude..."):
        result = answer(question, cat, k)
    colour = {"VERIFIED": "green", "NO MATCH": "orange", "UNVERIFIED": "red"}[result["status"]]
    st.markdown(f":{colour}[**{result['status']}**] {result['reason']}")
    st.markdown(result["answer"])
    st.subheader("Products retrieved")
    st.dataframe(pd.DataFrame([{"ID": h["id"], "cited": h["id"] in result["cited"],
                                "exact match": h["exact"], "similarity": round(h["score"], 3),
                                "details": h["doc"]} for h in result["retrieved"]]),
                 hide_index=True)
