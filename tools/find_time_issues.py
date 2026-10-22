import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
from music_reader.backends import ClassicalBackend, RecognitionOptions
from music_reader.pdfinput import Document
from music_reader.assemble import infer_meter

def check_hymn(d_str: str) -> dict | None:
    d = Path(d_str)
    score_p = d / "score.png"
    if not score_p.exists():
        return None
    try:
        backend = ClassicalBackend()
        doc = Document(score_p)
        res = backend.recognize(doc, RecognitionOptions())
        score = res.score
        meter = str(score.meter) if score.meter else None
        meter_src = getattr(score, "meter_source", None)
        
        diags = [diag for diag in score.diagnostics if "TIME_SIGNATURE" in diag.code]
        bars = infer_meter(score.melody.measures) if score.melody else None
        bars_str = str(bars) if bars else None
        
        overfull = [diag for diag in score.diagnostics if diag.code == "MEASURE_OVERFULL"]
        incomplete = [diag for diag in score.diagnostics if diag.code == "MEASURE_INCOMPLETE"]
        
        is_mismatch = bool(diags) or (bars and score.meter and bars.measure_duration != score.meter.measure_duration)
        is_unusual = score.meter and (score.meter.denominator not in (2, 4, 8) or score.meter.numerator not in (2, 3, 4, 6, 9, 12))
        
        if is_mismatch or is_unusual or len(overfull) >= 3:
            return {
                "hymn": d.name,
                "meter": meter,
                "meter_src": meter_src,
                "bars_meter": bars_str,
                "diags": [f"{diag.code}: {diag.message}" for diag in diags],
                "overfull_count": len(overfull),
                "incomplete_count": len(incomplete),
                "total_measures": len(score.melody.measures) if score.melody else 0,
                "is_unusual": is_unusual,
                "is_mismatch": is_mismatch,
            }
    except Exception as e:
        return {"hymn": d.name, "error": str(e)}
    return None

def main():
    vdir = Path(r"C:\Users\jsilv\musicus-data\sources\virsikirja")
    dirs = sorted([str(d) for d in vdir.iterdir() if d.is_dir() and (d / "score.png").exists()])
    print(f"Checking {len(dirs)} hymns with 8 workers...")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(check_hymn, dirs))
    elapsed = time.perf_counter() - t0
    issues = [r for r in results if r is not None and "error" not in r]
    errors = [r for r in results if r is not None and "error" in r]
    print(f"Done in {elapsed:.1f}s. Found {len(issues)} hymns with time issues, {len(errors)} errors.")
    with open("time_signature_issues.json", "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2)

if __name__ == "__main__":
    main()
