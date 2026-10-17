"""Sentence embeddings of distinct project or vendor names on the GPU (foundation for papers 4, 5, 6, 8).

  python foundation/fdn_embed.py --set vendor_names --model bge-m3
  python foundation/fdn_embed.py --set project_names --model e5-large --limit 5000 --out /tmp/x   # smoke test

Reads  $FDN/<set>.parquet (column `name`, row order = id order) made by fdn_tables.py names.
Writes $FDN/emb/<model>/<set>/shard_NNNNN.npy  float16, L2-normalised, rows in id order, SHARD rows each,
       then manifest.json (model, revision, dim, rows, versions, seconds). Finished shards are skipped,
       so a killed job resumes at the next shard.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

EGP = Path(os.environ.get("EGP", str(Path.home() / "egp"))).expanduser()
MODELS = {  # key: (HuggingFace id, text prefix)
    "bge-m3": ("BAAI/bge-m3", ""),
    "e5-large": ("intfloat/multilingual-e5-large", "query: "),
    "qwen3-emb-0.6b": ("Qwen/Qwen3-Embedding-0.6B", ""),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["project_names", "vendor_names"])
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--fdn", default=str(EGP / "data" / "shared" / "foundation_v1"))
    ap.add_argument("--out", help="default: <fdn>/emb/<model>/<set>")
    ap.add_argument("--shard", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=None, help="tokens; default 128 for projects, 64 for vendors")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available (check CUDA_VISIBLE_DEVICES and the egp-gpu env)")
    hf_id, prefix = MODELS[a.model]
    max_len = a.max_len or (128 if a.set == "project_names" else 64)
    out = Path(a.out) if a.out else Path(a.fdn) / "emb" / a.model / a.set
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    names = pq.read_table(Path(a.fdn) / f"{a.set}.parquet", columns=["name"]).column("name").to_pylist()
    if a.limit:
        names = names[:a.limit]
    n = len(names)
    print(f"{n:,} names from {a.set}; model {hf_id}; max_len {max_len}; out {out}", flush=True)

    model = SentenceTransformer(hf_id, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = max_len
    dim = model.get_sentence_embedding_dimension()
    print(f"loaded in {time.time() - t0:.0f} s; dim {dim}; GPU {torch.cuda.get_device_name(0)}", flush=True)

    n_shards = (n + a.shard - 1) // a.shard
    for i in range(n_shards):
        f = out / f"shard_{i:05d}.npy"
        if f.exists():
            continue
        ts = time.time()
        chunk = [prefix + (s or "") for s in names[i * a.shard:(i + 1) * a.shard]]
        emb = model.encode(chunk, batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False).astype(np.float16)
        if not np.isfinite(emb).all():
            raise SystemExit(f"non-finite values in shard {i}")
        tmp = out / f"shard_{i:05d}.tmp.npy"
        np.save(tmp, emb)
        os.replace(tmp, f)
        rate = len(chunk) / (time.time() - ts)
        left = (n - (i + 1) * a.shard) / rate if rate else 0
        print(f"shard {i + 1}/{n_shards}: {len(chunk):,} rows, {rate:,.0f} rows/s, "
              f"about {max(left, 0) / 3600:.1f} h left", flush=True)

    rev = getattr(getattr(model[0], "auto_model", None), "config", None)
    info = {"set": a.set, "model": hf_id, "prefix": prefix, "max_len": max_len, "dim": dim, "rows": n,
            "shard_rows": a.shard, "shards": n_shards, "dtype": "float16", "normalized": True,
            "revision": getattr(rev, "_commit_hash", None), "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "seconds": round(time.time() - t0),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "manifest.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
