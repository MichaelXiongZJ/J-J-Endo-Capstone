"""Score detected events against staged-clip ground truth (implementation guide §10).

§10 defines the metric but ships no code for it, so this implements it exactly as
written: match PER EVENT, not per frame, by time-window overlap.

    TP = a real violation that was flagged
    FP = a flag with no violation
    FN = a violation that was missed
    Precision = TP / (TP + FP)      Recall = TP / (TP + FN)

PoC target is precision >= 0.8 on Rules 3 and 5. Precision beats recall at this
stage: a system that cries wolf gets muted within a week and is then worse than
nothing (context.md §7.6).

Ground truth is one JSON file listing the intervals you noted while filming:

    {
      "clips": [
        {
          "video": "lean_out_01.mp4",
          "violations": [
            {"rule": 5, "start_s": 4.0, "end_s": 7.5, "note": "driver leans out left"}
          ]
        },
        {
          "video": "reversing_headturn_01.mp4",
          "violations": []
        }
      ]
    }

A clip with `"violations": []` is a NEGATIVE clip. These are the most valuable
ones you can film — especially the driver turning their head to look behind
while reversing, which is correct behaviour that naive Rule 5 logic wrongly
flags (context.md §6.3). Every event on a negative clip is an FP.

Usage:
    python -m scripts.score_events --events outputs/events/events.jsonl \
                                   --truth data/validation/ground_truth.json
"""

import argparse
import json
import os
from collections import defaultdict


def load_events(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def overlaps(event, gt, tolerance_s):
    """True if the event's time span overlaps the ground-truth interval.

    Tolerance is applied to the ground-truth interval rather than the event
    because every rule has a duration gate (R5_MIN_S etc.), so a detection
    legitimately lands ~1-2 s AFTER the violation begins. Without tolerance a
    correct detection of a short violation would score as FP + FN.
    """
    lo = gt['start_s'] - tolerance_s
    hi = gt['end_s'] + tolerance_s
    e_start = event.get('start_s', event.get('timestamp_s'))
    e_end = event.get('end_s', e_start)
    return e_start <= hi and e_end >= lo


def score(events, truth, tolerance_s=2.0, rules=None):
    """Returns (per_rule_counts, unmatched_events, unmatched_truths)."""
    by_video = defaultdict(list)
    for e in events:
        by_video[e.get('video', '')].append(e)

    counts = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    fps, fns = [], []

    for clip in truth['clips']:
        video = clip['video']
        clip_events = by_video.get(video, [])
        gts = clip.get('violations', [])
        if rules:
            clip_events = [e for e in clip_events if e['rule'] in rules]
            gts = [g for g in gts if g['rule'] in rules]

        matched_events = set()
        for gi, gt in enumerate(gts):
            # All events overlapping this interval count as ONE TP. Several
            # events for one violation is a fragmentation problem (raise
            # EventAggregator's cooldown), not several successes.
            hits = [i for i, e in enumerate(clip_events)
                    if i not in matched_events
                    and e['rule'] == gt['rule']
                    and overlaps(e, gt, tolerance_s)]
            if hits:
                counts[gt['rule']]['TP'] += 1
                matched_events.update(hits)
            else:
                counts[gt['rule']]['FN'] += 1
                fns.append({'video': video, **gt})

        for i, e in enumerate(clip_events):
            if i not in matched_events:
                counts[e['rule']]['FP'] += 1
                fps.append(e)

    return counts, fps, fns


def report(counts, fps, fns, targets=(3, 5), min_precision=0.8):
    rows = []
    tot = {'TP': 0, 'FP': 0, 'FN': 0}
    for rule in sorted(counts):
        c = counts[rule]
        for k in tot:
            tot[k] += c[k]
        p = c['TP'] / (c['TP'] + c['FP']) if (c['TP'] + c['FP']) else float('nan')
        r = c['TP'] / (c['TP'] + c['FN']) if (c['TP'] + c['FN']) else float('nan')
        rows.append((rule, c['TP'], c['FP'], c['FN'], p, r))

    print(f"{'Rule':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8}")
    print('-' * 42)
    for rule, tp, fp, fn, p, r in rows:
        print(f'{rule:>5} {tp:>4} {fp:>4} {fn:>4} {p:>10.3f} {r:>8.3f}')
    tp, fp, fn = tot['TP'], tot['FP'], tot['FN']
    p = tp / (tp + fp) if (tp + fp) else float('nan')
    r = tp / (tp + fn) if (tp + fn) else float('nan')
    print('-' * 42)
    print(f"{'ALL':>5} {tp:>4} {fp:>4} {fn:>4} {p:>10.3f} {r:>8.3f}")

    print(f'\nPoC gate: precision >= {min_precision} on rules {list(targets)}')
    verdict = True
    for rule in targets:
        c = counts.get(rule)
        if not c or (c['TP'] + c['FP']) == 0:
            print(f'  Rule {rule}: NO DATA — stage clips for this rule before reporting.')
            verdict = False
            continue
        pr = c['TP'] / (c['TP'] + c['FP'])
        ok = pr >= min_precision
        verdict &= ok
        print(f"  Rule {rule}: precision {pr:.3f}  [{'PASS' if ok else 'FAIL'}]")
    if not verdict:
        print('\nBelow target. Tune CFG in src/rules.py — raise the *_MIN_S duration '
              'gates first (they cost little recall), then tighten thresholds. '
              'Re-run the clips after each change; do NOT retrain.')

    if fps:
        print(f'\nFalse positives ({len(fps)}) — read these before tuning; the pattern '
              'usually names the fix:')
        for e in fps[:15]:
            extra = {k: v for k, v in e.items()
                     if k in ('distance_m', 'seconds_outside', 'floor_xy',
                              'person_track', 'driver_track')}
            print(f"  rule {e['rule']} @ {e.get('start_s')}-{e.get('end_s')}s "
                  f"in {e.get('video')}  {extra}")
        if len(fps) > 15:
            print(f'  ... and {len(fps) - 15} more')

    if fns:
        print(f'\nMissed violations ({len(fns)}):')
        for g in fns[:15]:
            print(f"  rule {g['rule']} @ {g['start_s']}-{g['end_s']}s in {g['video']}"
                  f"  {g.get('note', '')}")
        if len(fns) > 15:
            print(f'  ... and {len(fns) - 15} more')

    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--events', required=True, help='events.jsonl from run_pipeline')
    ap.add_argument('--truth', required=True, help='ground-truth JSON (format in docstring)')
    ap.add_argument('--tolerance', type=float, default=2.0,
                    help='seconds of slack around each ground-truth interval, to allow '
                         'for the rules\' duration gates (default 2.0)')
    ap.add_argument('--rules', type=int, nargs='*', default=None,
                    help='restrict scoring to these rules')
    ap.add_argument('--min-precision', type=float, default=0.8)
    args = ap.parse_args()

    if not os.path.exists(args.events):
        raise SystemExit(f'no events file at {args.events} — run src.run_pipeline first')
    events = load_events(args.events)
    with open(args.truth) as f:
        truth = json.load(f)

    n_clips = len(truth['clips'])
    n_gt = sum(len(c.get('violations', [])) for c in truth['clips'])
    n_neg = sum(1 for c in truth['clips'] if not c.get('violations'))
    print(f'{len(events)} events | {n_clips} clips ({n_neg} negative) | '
          f'{n_gt} ground-truth violations | tolerance {args.tolerance}s\n')
    if n_clips < 20:
        print(f'NOTE: §10 calls for 20-30 staged clips; you have {n_clips}. '
              'Numbers from fewer clips are indicative, not reportable.\n')
    if n_neg == 0:
        print('WARNING: no negative clips. Without them precision is unmeasurable — '
              'especially the reversing head-turn case (context.md §6.3).\n')

    counts, fps, fns = score(events, truth, args.tolerance,
                             set(args.rules) if args.rules else None)
    ok = report(counts, fps, fns, min_precision=args.min_precision)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
