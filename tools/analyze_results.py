import json

with open("virsikirja_comparison.json", "r", encoding="utf-8") as f:
    data = json.load(f)

evaluated = [r for r in data if r.get("status") == "ok"]
errors = [r for r in data if r.get("status") == "error"]
skipped = [r for r in data if r.get("status") == "skipped"]

print(f"Total entries in dataset: {len(data)}")
print(f"Evaluated (both audio & score): {len(evaluated)}")
print(f"Skipped: {len(skipped)}")
print(f"Exceptions: {len(errors)}")

under_03 = [r for r in evaluated if r["similarity"] < 0.3]
under_05 = [r for r in evaluated if 0.3 <= r["similarity"] < 0.5]
between_05_07 = [r for r in evaluated if 0.5 <= r["similarity"] < 0.7]
between_07_08 = [r for r in evaluated if 0.7 <= r["similarity"] < 0.8]
between_08_09 = [r for r in evaluated if 0.8 <= r["similarity"] < 0.9]
above_09 = [r for r in evaluated if r["similarity"] >= 0.9]

print("\n--- Similarity Distribution ---")
print(f"  < 0.30:    {len(under_03):3d}  ({len(under_03)/len(evaluated)*100:4.1f}%)")
print(f"  0.30-0.50: {len(under_05):3d}  ({len(under_05)/len(evaluated)*100:4.1f}%)")
print(f"  0.50-0.70: {len(between_05_07):3d}  ({len(between_05_07)/len(evaluated)*100:4.1f}%)")
print(f"  0.70-0.80: {len(between_07_08):3d}  ({len(between_07_08)/len(evaluated)*100:4.1f}%)")
print(f"  0.80-0.90: {len(between_08_09):3d}  ({len(between_08_09)/len(evaluated)*100:4.1f}%)")
print(f"  >= 0.90:   {len(above_09):3d}  ({len(above_09)/len(evaluated)*100:4.1f}%)")

print(f"\n--- ALL hymns under 0.30 (Count: {len(under_03)}) ---")
for r in under_03:
    print(f"  {r['hymn']}: sim={r['similarity']} audio_n={r['audio_notes']} score_n={r['score_notes']} key={r['key_fifths']:+d} errs={r['errors']} warns={r['warnings']}")

print(f"\n--- Lowest 15 hymns between 0.30 and 0.50 ---")
for r in sorted(under_05, key=lambda x: x["similarity"])[:15]:
    print(f"  {r['hymn']}: sim={r['similarity']} audio_n={r['audio_notes']} score_n={r['score_notes']} key={r['key_fifths']:+d} errs={r['errors']}")
