"""Per-frame rule hits -> one event per real-world episode.

NOT IN THE IMPLEMENTATION GUIDE. Added because the guide's §9 pipeline writes one
JSONL line AND one evidence JPEG for every processed frame in which a rule
condition holds. A pedestrian standing 5 m from a forklift for 20 seconds
produces 200 "events" and 200 near-identical JPEGs.

That breaks two things the project depends on:

  * §10 scores precision/recall PER EVENT. With per-frame emission one true
    violation counts as 200 TPs, so measured precision is meaningless.
  * Evidence frames are the human-review artifact. 200 copies of one moment is
    not reviewable.

So hits are grouped into episodes: a (rule, participants) key stays open while
hits keep arriving, closes after COOLDOWN_S of silence, and emits one record with
start/end/duration plus the peak severity. The evidence frame is taken at the
peak, not the first frame — the most severe instant is what a reviewer wants.

Rule logic in src/rules.py is untouched; only counting and storage change.
"""

import json
import os

import cv2

# A lapse shorter than this is one continuous episode, not two. Covers momentary
# detection dropouts and occlusion flicker.
COOLDOWN_S = 2.0

# Which fields identify an episode, per rule. Everything else in a hit dict is
# treated as a per-frame measurement.
KEY_FIELDS = {1: ('person_track',),
              3: ('person_track', 'vehicle_track'),
              4: ('person_track',),
              5: ('driver_track',)}

# Peak severity per rule: Rule 3's is the CLOSEST approach, Rule 5's the LONGEST
# lean-out. Rules 1 and 4 are binary, so they have no severity measure.
SEVERITY = {3: ('distance_m', min),
            5: ('seconds_outside', max)}


class EventAggregator:
    """Collapses per-frame rule hits into per-episode events.

        agg = EventAggregator('outputs/events', 'cam1', 'clip.mp4')
        for each processed frame:
            agg.add(hits, t, bgr)
        agg.close()
    """

    def __init__(self, outdir, camera_id, video, cooldown_s=COOLDOWN_S,
                 save_frames=True):
        os.makedirs(outdir, exist_ok=True)
        self.dir = outdir
        self.camera_id = camera_id
        self.video = video
        self.cooldown_s = cooldown_s
        self.save_frames = save_frames
        self.path = os.path.join(outdir, 'events.jsonl')
        self._f = open(self.path, 'w')      # one run = one events file
        self.open_eps = {}                  # key -> episode dict
        self.count = 0
        self.active_rules = set()           # for the on-screen banner

    def add(self, hits, t, frame=None):
        """Feed one frame's hits. Returns the set of rules currently active."""
        for key in [k for k, ep in self.open_eps.items()
                    if t - ep['last_t'] > self.cooldown_s]:
            self._emit(self.open_eps.pop(key))

        self.active_rules = set()
        for hit in hits:
            rule = hit['rule']
            self.active_rules.add(rule)
            key = (rule,) + tuple(hit.get(f) for f in KEY_FIELDS.get(rule, ()))
            ep = self.open_eps.get(key)
            if ep is None:
                self.open_eps[key] = {'rule': rule, 'start_t': t, 'last_t': t,
                                      'peak_t': t, 'frames': 1, 'hit': dict(hit),
                                      'frame': frame}
            else:
                ep['last_t'] = t
                ep['frames'] += 1
                self._maybe_new_peak(ep, t, hit, frame)
        return self.active_rules

    @staticmethod
    def _maybe_new_peak(ep, t, hit, frame):
        field = SEVERITY.get(ep['rule'])
        if field is None:
            return                          # binary rule: first frame stays as peak
        name, better = field
        new, old = hit.get(name), ep['hit'].get(name)
        if new is not None and (old is None or better(new, old) == new):
            ep.update(hit=dict(hit), peak_t=t, frame=frame)

    def close(self):
        for key in list(self.open_eps):
            self._emit(self.open_eps.pop(key))
        self._f.close()
        return self.count

    def _emit(self, ep):
        rec = dict(ep['hit'])
        rec.update({
            'event_id': f'evt_{self.count:05d}',
            'camera_id': self.camera_id,
            'video': self.video,
            'start_s': round(ep['start_t'], 2),
            'end_s': round(ep['last_t'], 2),
            'duration_s': round(ep['last_t'] - ep['start_t'], 2),
            'peak_s': round(ep['peak_t'], 2),
            'frames': ep['frames'],
            # Kept so anything reading the guide's original single-timestamp
            # format still works.
            'timestamp_s': round(ep['peak_t'], 2),
        })
        if self.save_frames and ep['frame'] is not None:
            p = os.path.join(self.dir, f"{rec['event_id']}_rule{ep['rule']}.jpg")
            cv2.imwrite(p, ep['frame'])
            rec['evidence_frame'] = p
        self._f.write(json.dumps(rec) + '\n')
        self._f.flush()
        self.count += 1
