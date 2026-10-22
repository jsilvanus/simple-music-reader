import json
import numpy as np

with open("virsikirja_comparison.json", "r", encoding="utf-8") as f:
    data = json.load(f)

evaluated = [r for r in data if r.get("status") == "ok"]
errors = [r for r in data if r.get("status") == "error"]
skipped = [r for r in data if r.get("status") == "skipped"]

print(f"Total entries in dataset: {len(data)}")
print(f"Evaluated (both audio & score): {len(evaluated)}")
print(f"Skipped: {len(skipped)}")
print(f"Exceptions: {len(errors)}")

sims = [r["similarity"] for r in evaluated]
print(f"\nMean similarity: {np.mean(sims):.3f}")
print(f"Median similarity: {np.median(sims):.3f}")

# Group by 0.10 gaps
bins = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.001),
]

print("\n--- Similarity Distribution (0.10 Bins) ---")
for lo, hi in bins:
    label = f"{lo:.2f} - {1.00 if hi > 1.0 else hi:.2f}"
    count = sum(1 for s in sims if lo <= s < hi) if hi <= 1.0 else sum(1 for s in sims if lo <= s <= 1.0)
    pct = count / len(evaluated) * 100
    bar = "#" * int(round(pct / 2))
    print(f"  {label}: {count:3d} ({pct:5.1f}%) | {bar}")

under_03 = [r for r in evaluated if r["similarity"] < 0.3]
print(f"\n--- Hymns under 0.30 (Count: {len(under_03)}) ---")
for r in sorted(under_03, key=lambda x: x["similarity"]):
    print(f"  {r['hymn']}: sim={r['similarity']} audio_n={r['audio_notes']} score_n={r['score_notes']} key={r['key_fifths']:+d} errs={r['errors']}")

under_05 = [r for r in evaluated if 0.30 <= r["similarity"] < 0.50]
print(f"\n--- Lowest 10 hymns between 0.30 and 0.50 (Count: {len(under_05)}) ---")
for r in sorted(under_05, key=lambda x: x["similarity"])[:10]:
    print(f"  {r['hymn']}: sim={r['similarity']} audio_n={r['audio_notes']} score_n={r['score_notes']} key={r['key_fifths']:+d} errs={r['errors']}")
