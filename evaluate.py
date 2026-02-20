import os
import sys
import csv
import pickle
from pathlib import Path
from collections import defaultdict

from models.damerau_levenshtein import (
    similarity_score, normalize,
    compute_wer, compute_cer, find_best_ocr_token,
)
from models.drug_graph import build_drug_graph

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def predict_label(ocr_texts, labels, graph=None, top_k=3,
                  conflict_threshold=0.55):

    if not ocr_texts:
        return "UNKNOWN", 0.0, "no_ocr"

    label_scores = {}
    for label in labels:
        label_norm = normalize(label)
        best = 0.0
        for token in ocr_texts:
            token_norm = normalize(token)
            if not token_norm:
                continue
            score = similarity_score(token_norm, label_norm)
            if score > best:
                best = score
        label_scores[label] = best

    ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return "UNKNOWN", 0.0, "no_ocr"

    best_label, best_score = ranked[0]

    if graph is not None and best_score >= conflict_threshold:
        top_candidates = [
            lbl for lbl, sc in ranked[:top_k]
            if sc >= conflict_threshold
        ]
        if len(top_candidates) >= 2:
            resolved = graph.resolve_conflict(top_candidates)
            if resolved is not None:
                return resolved, label_scores[resolved], "graph_brand_rule"

    return best_label, round(best_score, 4), "dl_best"



def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/").lower()


def build_pickle_lookup(sentences_dict):
    return {normalize_path(k): v for k, v in sentences_dict.items()}


def show_evaluation(results, output_csv):
    skip_labels = {'FILE_NOT_FOUND', 'OCR_ERROR', 'OCR_NOT_CACHED', 'UNKNOWN'}
    valid = [r for r in results if r['predicted_label'] not in skip_labels]
    n = len(valid)
    skipped = len(results) - n

    if n == 0:
        print("Tidak ada prediksi valid untuk dievaluasi.")
        return

    correct_count = sum(1 for r in valid if r['correct'] is True)
    accuracy = correct_count / n * 100

    total_wer_dl = sum(compute_wer(r['true_label'], r['predicted_label']) for r in valid)
    total_cer_dl = sum(compute_cer(r['true_label'], r['predicted_label']) for r in valid)
    avg_wer_dl = total_wer_dl / n * 100
    avg_cer_dl = total_cer_dl / n * 100

    total_wer_raw = 0.0
    total_cer_raw = 0.0
    raw_data = []
    for r in valid:
        best_token = find_best_ocr_token(r.get('ocr_text', ''), r['true_label'])
        wer_raw = compute_wer(r['true_label'], best_token)
        cer_raw = compute_cer(r['true_label'], best_token)
        total_wer_raw += wer_raw
        total_cer_raw += cer_raw
        raw_data.append({
            'true_label': r['true_label'],
            'wer_raw': wer_raw,
            'cer_raw': cer_raw,
        })
    avg_wer_raw = total_wer_raw / n * 100
    avg_cer_raw = total_cer_raw / n * 100

    # Per-label stats
    per_label = defaultdict(lambda: {
        'total': 0, 'correct': 0,
        'wer_raw': 0.0, 'cer_raw': 0.0,
        'wer_dl': 0.0, 'cer_dl': 0.0,
    })
    for r, raw in zip(valid, raw_data):
        lbl = r['true_label']
        per_label[lbl]['total'] += 1
        if r['correct'] is True:
            per_label[lbl]['correct'] += 1
        per_label[lbl]['wer_raw'] += raw['wer_raw']
        per_label[lbl]['cer_raw'] += raw['cer_raw']
        per_label[lbl]['wer_dl'] += compute_wer(r['true_label'], r['predicted_label'])
        per_label[lbl]['cer_dl'] += compute_cer(r['true_label'], r['predicted_label'])

    graph_resolved = sum(1 for r in valid if r.get('resolution') == 'graph_brand_rule')

    print("\n" + "=" * 75)
    print("                       EVALUATION SUMMARY")
    print("=" * 75)
    print(f"  CSV File              : {output_csv}")
    print(f"  Total rows            : {len(results)}")
    print(f"  Skipped (error/none)  : {skipped}")
    print(f"  Valid predictions      : {n}")
    print(f"  Correct predictions    : {correct_count}")
    print(f"  Accuracy               : {accuracy:.2f}%")
    print(f"  Graph brand-rule used  : {graph_resolved} kali")
    print()
    print("  " + "-" * 55)
    print(f"  {'Metrik':<28} {'Raw OCR':>12} {'Setelah DL':>12}")
    print("  " + "-" * 55)
    print(f"  {'Average WER':<28} {avg_wer_raw:>11.2f}% {avg_wer_dl:>11.2f}%")
    print(f"  {'Average CER':<28} {avg_cer_raw:>11.2f}% {avg_cer_dl:>11.2f}%")
    print("  " + "-" * 55)
    print()
    print("  Keterangan:")
    print("    Raw OCR    = token OCR mentah terdekat vs true_label")
    print("    Setelah DL = predicted_label (hasil DL matching) vs true_label")
    print()

    print("Per-Label Breakdown:")
    print(f"  {'Label':<16} {'Acc%':>6} | {'WER_raw':>8} {'CER_raw':>8} | {'WER_dl':>8} {'CER_dl':>8}")
    print("  " + "-" * 65)
    for label in sorted(per_label.keys()):
        d = per_label[label]
        t = d['total']
        acc = d['correct'] / t * 100 if t else 0
        wr = d['wer_raw'] / t * 100 if t else 0
        cr = d['cer_raw'] / t * 100 if t else 0
        wd = d['wer_dl'] / t * 100 if t else 0
        cd = d['cer_dl'] / t * 100 if t else 0
        print(f"  {label:<16} {acc:>5.1f}% | {wr:>7.2f}% {cr:>7.2f}% | {wd:>7.2f}% {cd:>7.2f}%")
    print("=" * 75)

    wrong = [r for r in valid if r['correct'] is not True]
    if wrong:
        print(f"\nSample Salah Prediksi ({min(10, len(wrong))} dari {len(wrong)}):")
        print(f"  {'Gambar':<35} {'True':<16} {'Pred':<16} Score")
        print("  " + "-" * 75)
        for r in wrong[:10]:
            name = r['image_path'].split('\\')[-1][:33]
            print(f"  {name:<35} {r['true_label']:<16} {r['predicted_label']:<16} {r['best_score']}")

    print(f"\nDone! Full results saved to: {output_csv}")


def main():
    DB_CSV      = Path("dataset.csv")
    OUTPUT_CSV  = Path("prediction_results.csv")
    PICKLE_PATH = Path("output/ocr_results.pkl")

    if not PICKLE_PATH.exists():
        print(f"ERROR: Pickle file '{PICKLE_PATH}' not found!")
        print("Jalankan `python train.py` terlebih dahulu.")
        return

    print(f"Loading OCR results from: {PICKLE_PATH} ...")
    with open(PICKLE_PATH, "rb") as f:
        pickle_data = pickle.load(f)

    pkl_images    = pickle_data.get("images", {})
    pkl_sentences = pickle_data.get("sentences", {})
    print(f"  Loaded {len(pkl_sentences)} OCR text results")
    print(f"  Loaded {len(pkl_images)} result images")

    sentences_lookup = build_pickle_lookup(pkl_sentences)

    print(f"\nLoading {DB_CSV} ...")
    db_rows = []
    labels_set = set()

    with open(DB_CSV, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = 'Image Name' if 'Image Name' in row else list(row.keys())[0]
            img_name = row[key].strip()
            label    = row['Label'].strip()
            db_rows.append((img_name, label))
            labels_set.add(label)

    unique_labels = sorted(labels_set)
    print(f"  Total images : {len(db_rows)}")
    print(f"  Unique labels: {len(unique_labels)} → {unique_labels}")

    drug_graph = build_drug_graph()
    print(f"\n{drug_graph.summary()}\n")

    results = []
    cache_hits = 0
    cache_misses = 0

    print("=" * 70)
    print(f"{'#':<5} {'Image':<40} {'True':<16} {'Pred':<16} {'Score':<7}")
    print("=" * 70)

    for idx, (img_name, true_label) in enumerate(db_rows, 1):
        norm_key = normalize_path(img_name)

        ocr_texts = sentences_lookup.get(norm_key)

        if ocr_texts is None:
            cache_misses += 1
            print(f"{idx:<5} [SKIP - not in pickle] {img_name}")
            results.append({
                'image_path'     : img_name,
                'true_label'     : true_label,
                'predicted_label': 'OCR_NOT_CACHED',
                'best_score'     : 0.0,
                'resolution'     : '',
                'ocr_text'       : '',
                'correct'        : False
            })
            continue

        cache_hits += 1

        pred_label, score, resolution = predict_label(
            ocr_texts, unique_labels, graph=drug_graph
        )
        correct = (pred_label.lower() == true_label.lower())
        ocr_combined = ' | '.join(ocr_texts)

        display_name = img_name.split('\\')[-1] if '\\' in img_name else img_name
        res_tag = '[G]' if resolution == 'graph_brand_rule' else '   '
        mark    = 'OK' if correct else 'XX'
        print(f"{idx:<5} {display_name:<38} {true_label:<16} {pred_label:<16} {score:<7.4f} {mark} {res_tag}")

        results.append({
            'image_path'     : img_name,
            'true_label'     : true_label,
            'predicted_label': pred_label,
            'best_score'     : score,
            'resolution'     : resolution,
            'ocr_text'       : ocr_combined,
            'correct'        : correct
        })

    print(f"\nSaving {OUTPUT_CSV} ...")
    fieldnames = ['image_path', 'true_label', 'predicted_label', 'best_score',
                  'resolution', 'ocr_text', 'correct']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved {len(results)} rows")
    print(f"  Pickle cache: {cache_hits} hits, {cache_misses} misses")

    show_evaluation(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()
