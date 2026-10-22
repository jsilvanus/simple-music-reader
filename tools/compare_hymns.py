"""Compare simple-music-reader score recognition against liturgos-musicus audio pitch tracking."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

MUSICUS_SRC = Path(r"c:\Users\jsilv\Code\liturgos-musicus\src")
if str(MUSICUS_SRC) not in sys.path:
    sys.path.insert(0, str(MUSICUS_SRC))

from musicus.audio import decode
from musicus.pitch import yin, to_notes
from musicus.melody import pitch_steps, subsequence_dtw, similarity
from music_reader.backends import get_backend, RecognitionOptions
from music_reader.pdfinput import Document


def eval_hymn(hymn_path_str: str) -> dict:
    hymn_dir = Path(hymn_path_str)
    score_p = hymn_dir / "score.png"
    audio_files = list((hymn_dir / "audio").glob("melody-*.mp3")) if (hymn_dir / "audio").exists() else []
    if not score_p.exists() or not audio_files:
        return {"hymn": hymn_dir.name, "status": "skipped", "reason": "missing score or audio"}
    
    t0 = time.perf_counter()
    try:
        # Audio pitch sequence via musicus
        y = decode(audio_files[0])
        a_track = yin(y)
        a_notes = to_notes(a_track)
        a_midis = [round(n.midi) for n in a_notes]
        a_steps = pitch_steps(a_midis)
        
        # Score notation via simple-music-reader
        backend = get_backend("classical")
        doc = Document(score_p)
        res = backend.recognize(doc, RecognitionOptions())
        s_notes = [e for e in res.score.melody.events() if e.kind == "note"]
        s_midis = [e.pitch.midi for e in s_notes]
        s_steps = pitch_steps(s_midis)
        
        # Sequence matching
        cost, i0, i1 = subsequence_dtw(s_steps, a_steps) if len(s_steps) and len(a_steps) else (float("inf"), 0, 0)
        sim = similarity(s_steps, a_steps) if len(s_steps) and len(a_steps) else 0.0
        elapsed = time.perf_counter() - t0
        
        warns = [d.code for d in res.score.diagnostics if d.severity == "warning"]
        errs = [d.code for d in res.score.diagnostics if d.severity == "error"]
        
        return {
            "hymn": hymn_dir.name,
            "status": "ok",
            "audio_notes": len(a_midis),
            "score_notes": len(s_midis),
            "key_fifths": res.score.key.fifths if res.score.key else 0,
            "similarity": round(sim, 3),
            "cost": round(cost, 3),
            "errors": errs,
            "warnings": warns,
            "elapsed_s": round(elapsed, 2)
        }
    except Exception as e:
        return {
            "hymn": hymn_dir.name,
            "status": "error",
            "error": str(e),
            "elapsed_s": round(time.perf_counter() - t0, 2)
        }


def main():
    parser = argparse.ArgumentParser(description="Compare music reader notes against musicus audio pitches")
    parser.add_argument("--data-dir", default=r"C:\Users\jsilv\musicus-data\sources\virsikirja")
    parser.add_argument("--count", type=int, default=0, help="Number of hymns to evaluate (0 for all)")
    parser.add_argument("--workers", type=int, default=6, help="Worker processes")
    parser.add_argument("--out", help="Write JSON results to file")
    args = parser.parse_args()

    vdir = Path(args.data_dir)
    dirs = sorted([d for d in vdir.iterdir() if d.is_dir()])
    if args.count > 0:
        dirs = dirs[:args.count]

    print(f"Comparing {len(dirs)} hymns with {args.workers} worker processes...")
    t0 = time.perf_counter()
    
    paths = [str(d) for d in dirs]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(eval_hymn, paths))
    else:
        results = [eval_hymn(p) for p in paths]
        
    total_time = time.perf_counter() - t0
    evaluated = [r for r in results if r.get("status") == "ok"]
    
    print(f"Completed {len(evaluated)} hymns in {total_time:.1f}s ({total_time / max(1, len(evaluated)):.2f}s per hymn).")
    
    if evaluated:
        avg_sim = sum(r["similarity"] for r in evaluated) / len(evaluated)
        print(f"Average similarity: {avg_sim:.3f}")
        
        # Hardest cases (lowest similarity)
        hard_cases = sorted(evaluated, key=lambda r: (r["similarity"], -len(r["errors"])))[:15]
        print("\n--- Lowest Similarity Cases (Hard Cases to investigate) ---")
        for h in hard_cases:
            print(f"  {h['hymn']}: sim={h['similarity']:.2f} cost={h['cost']:.2f} audio_n={h['audio_notes']} score_n={h['score_notes']} key={h['key_fifths']:+d} errs={len(h['errors'])} warns={len(h['warnings'])}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote full results to {args.out}")


if __name__ == "__main__":
    main()
