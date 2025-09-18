# script/measure_tokens.py
# Stima dei "costi" per modello in termini di token processati per claim:
# - retrieval@k: claim + tutti i doc recuperati (prediction/retrievals.jsonl)
# - evidence-only: claim + soli doc etichettati SUPPORTS/REFUTES dal modello
# Include confronto normalizzato e un grafico a barre su evidence tokens.

import json
import os
import sys
from collections import defaultdict

import pandas as pd

# ------------------ Percorsi dataset ------------------
DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
CLAIMS_FILE = os.path.join(DATA_DIR, "claims.jsonl")
RETR_FILE = os.path.join("prediction", "retrievals.jsonl")

# Il file nella release è "model_predictions.parqet" (senza 'u'); supportiamo anche parquet
PRED_FILE_CANDIDATES = [
    os.path.join("prediction", "model_predictions.parquet"),
    os.path.join("prediction", "model_predictions.parqet"),
]

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Nessun file predizioni trovato tra: {paths}")

PRED_FILE = first_existing(PRED_FILE_CANDIDATES)

# ------------------ Config tokenizer per modello ------------------
# Coerenti con il paper:
# - paragraph_joint / arsjoint -> roberta-base (512)
# - multivers_10 / multivers_20 -> longformer-base-4096 (4096)
# - vert5erini -> t5-base (512)
TOKENIZER_CFG = {
    "paragraph_joint": {"pretrained": "roberta-base", "max_len": 512},
    "arsjoint":        {"pretrained": "roberta-base", "max_len": 512},
    "multivers_10":    {"pretrained": "allenai/longformer-base-4096", "max_len": 4096},
    "multivers_20":    {"pretrained": "allenai/longformer-base-4096", "max_len": 4096},
    "vert5erini":      {"pretrained": "t5-base", "max_len": 512},
}

# ------------------ Tokenizer HF ------------------
from transformers import AutoTokenizer
TOKENIZERS = {
    m: AutoTokenizer.from_pretrained(cfg["pretrained"], use_fast=True)
    for m, cfg in TOKENIZER_CFG.items()
}

# ------------------ Util ------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def count_tokens(tokenizer, text, max_len):
    if not text:
        return 0
    ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len
    )["input_ids"]
    return len(ids)

# ------------------ Caricamento dati ------------------
print("Carico corpus, claims, retrievals, predictions...")
corpus = load_jsonl(CORPUS_FILE)
claims = load_jsonl(CLAIMS_FILE)
retr = load_jsonl(RETR_FILE)
pred = pd.read_parquet(PRED_FILE)

# mappa doc_id -> abstract
doc2abs = {}
for d in corpus:
    did = d.get("doc_id") or d.get("id")
    if did is not None:
        doc2abs[did] = d.get("abstract", "") or ""

# mappa claim_id -> testo claim
cid2claim = {}
for c in claims:
    cid = c.get("id") or c.get("claim_id")
    if cid is not None:
        cid2claim[cid] = c.get("claim", "")

# mappa claim_id -> lista doc_ids recuperati
cid2retr = {}
for r in retr:
    cid = r.get("claim_id") or r.get("id")
    docs = r.get("doc_ids") or r.get("docs") or []
    if cid is not None:
        cid2retr[cid] = docs

# ------------------ Normalizza colonne predizioni ------------------
pred.columns = [c.lower() for c in pred.columns]

def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

col_model = pick(pred, "model")
col_cid   = pick(pred, "claim_id", "cid", "claim")
col_did   = pick(pred, "doc_id", "did", "doc")
# includi predicted_label (è così nel file rilasciato)
col_lab   = pick(pred, "predicted_label", "label_pred", "pred_label", "prediction", "pred")

needed = [col_model, col_cid, col_did, col_lab]
if not all(needed):
    raise ValueError(f"Colonne non riconosciute in predictions: {pred.columns.tolist()}")

# normalizza etichette a {SUPPORTS, REFUTES, NEI}
def norm_label(x):
    x = str(x).upper()
    if x.startswith("SUPP"):
        return "SUPPORTS"
    if x.startswith("CONT") or x.startswith("REF"):
        return "REFUTES"
    return "NEI"

pred["label"] = pred[col_lab].map(norm_label)

# ------------------ Calcolo token ------------------
rows = []
print("Calcolo token per claim/modello...")

for model, cfg in TOKENIZER_CFG.items():
    tok = TOKENIZERS[model]
    max_len = cfg["max_len"]

    # predizioni di questo modello
    p_m = pred[pred[col_model] == model].copy()

    # doc non-NEI per claim (solo quelli giudicati evidence dal modello)
    non_nei = p_m[p_m["label"] != "NEI"].groupby(col_cid)[col_did].apply(list).to_dict()

    for cid, claim_text in cid2claim.items():
        # claim tokens
        claim_tokens = count_tokens(tok, claim_text, max_len)

        # Retrieval@k: tutti i doc recuperati per questo claim
        docs_k = cid2retr.get(cid, [])
        doc_tokens_sum = 0
        docs_count = 0
        for did in docs_k:
            abs_text = doc2abs.get(did, "")
            if abs_text:
                doc_tokens_sum += count_tokens(tok, abs_text, max_len)
                docs_count += 1

        # Evidence-only: solo doc che questo modello ha etichettato SUPPORTS/REFUTES
        docs_ev = non_nei.get(cid, [])
        ev_tokens_sum = 0
        ev_count = 0
        for did in docs_ev:
            abs_text = doc2abs.get(did, "")
            if abs_text:
                ev_tokens_sum += count_tokens(tok, abs_text, max_len)
                ev_count += 1

        rows.append({
            "claim_id": cid,
            "model": model,
            "claim_tokens": claim_tokens,
            "retrieved_docs": docs_count,
            "retrieved_doc_tokens_total": doc_tokens_sum,
            "retrieved_total_tokens": claim_tokens + doc_tokens_sum,
            "evidence_docs": ev_count,
            "evidence_doc_tokens_total": ev_tokens_sum,
            "evidence_total_tokens": claim_tokens + ev_tokens_sum,
        })

df = pd.DataFrame(rows)
out_csv = "tokens_per_claim_model.csv"
df.to_csv(out_csv, index=False)
print(f"Salvato: {out_csv}")

# ------------------ Riepilogo medio per modello + normalizzazione ------------------
summary = (
    df.groupby("model")[
        [
            "claim_tokens",
            "retrieved_docs",
            "retrieved_doc_tokens_total",
            "retrieved_total_tokens",
            "evidence_docs",
            "evidence_doc_tokens_total",
            "evidence_total_tokens",
        ]
    ]
    .mean()
    .round(2)
    .sort_values("evidence_total_tokens")  # ordina per costo evidence crescente
)

# normalizzazione (x volte il minimo)
base_ev = summary["evidence_total_tokens"].min()
summary["evidence_cost_rel"] = (summary["evidence_total_tokens"] / base_ev).round(2)

base_ret = summary["retrieved_total_tokens"].min()
summary["retrieved_cost_rel"] = (summary["retrieved_total_tokens"] / base_ret).round(2)

print("\n== Media per modello (per claim) ==")
print(summary)

summary_csv = "tokens_summary_by_model.csv"
summary.to_csv(summary_csv)
print(f"Salvato: {summary_csv}")

# ------------------ Grafico a barre (evidence tokens) ------------------
try:
    import matplotlib.pyplot as plt

    ax = summary["evidence_total_tokens"].plot(
        kind="bar",
        title="Evidence tokens per claim (media)",
        ylabel="token",
        xlabel="modello",
        rot=0
    )
    plt.tight_layout()
    out_png = "evidence_tokens_bar.png"
    plt.savefig(out_png)
    print(f"Salvato: {out_png}")

except ImportError:
    print("\n[INFO] matplotlib non è installato: salto il grafico.")
    print("       Per averlo:  python -m pip install matplotlib")
