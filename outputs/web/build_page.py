"""Build the shareable progress page, embedding images as data URIs.

The Artifact CSP blocks every external host, so images must be inlined. Written
as a generator rather than by hand because 400 KB of base64 in a source file is
unreviewable.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def img(name):
    with open(os.path.join(HERE, f'{name}.b64')) as f:
        return 'data:image/jpeg;base64,' + f.read()


HTML = """<title>Warehouse Safety CV — Capstone Progress</title>
<style>
:root {
  color-scheme: light dark;
  /* Concrete greys carry a blue bias so the neutral reads chosen, not default. */
  --ground:#EEF1F4; --surface:#FFFFFF; --sunk:#E3E8ED;
  --ink:#161B21; --muted:#5B6874; --line:#D2DAE2;
  --accent:#1F5F8B;          /* industrial blue, from the forklifts in our own footage */
  --ok:#2E7D52; --warn:#A8720F; --crit:#AC4028;   /* semantic only, never decorative */
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ground:#12161A; --surface:#191F25; --sunk:#0D1114;
    --ink:#E7ECF1; --muted:#93A2B0; --line:#2A333C;
    --accent:#6FB2DC; --ok:#5FBF8B; --warn:#D9A33F; --crit:#E2795C;
  }
}
:root[data-theme="dark"] {
  --ground:#12161A; --surface:#191F25; --sunk:#0D1114;
  --ink:#E7ECF1; --muted:#93A2B0; --line:#2A333C;
  --accent:#6FB2DC; --ok:#5FBF8B; --warn:#D9A33F; --crit:#E2795C;
}
:root[data-theme="light"] {
  --ground:#EEF1F4; --surface:#FFFFFF; --sunk:#E3E8ED;
  --ink:#161B21; --muted:#5B6874; --line:#D2DAE2;
  --accent:#1F5F8B; --ok:#2E7D52; --warn:#A8720F; --crit:#AC4028;
}

* { box-sizing:border-box; }
body {
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:920px; margin:0 auto; padding:48px 24px 96px; }
.prose { max-width:68ch; }

h1,h2,h3 { text-wrap:balance; line-height:1.18; margin:0; }
h1 { font-size:clamp(28px,4.4vw,40px); font-weight:680; letter-spacing:-.022em; }
h2 { font-size:22px; font-weight:650; letter-spacing:-.012em; margin:56px 0 4px; }
h3 { font-size:16px; font-weight:640; margin:28px 0 2px; }
p { margin:12px 0; }
a { color:var(--accent); }

.eyebrow {
  font-family:var(--mono); font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); margin-bottom:14px;
}
.lede { font-size:18.5px; color:var(--ink); margin-top:14px; }
.meta {
  font-family:var(--mono); font-size:12.5px; color:var(--muted);
  margin-top:22px; padding-top:16px; border-top:1px solid var(--line);
  display:flex; flex-wrap:wrap; gap:8px 22px;
}
.rule { height:1px; background:var(--line); border:0; margin:0; }

/* Status board: the stripe encodes state, so it reads before the words do. */
.board { display:flex; flex-direction:column; gap:2px; margin-top:20px; }
.row {
  display:grid; grid-template-columns:4px 1fr auto; gap:16px; align-items:center;
  background:var(--surface); border:1px solid var(--line); border-left:0;
  padding:13px 16px 13px 0; border-radius:3px;
}
.row .stripe { align-self:stretch; border-radius:3px 0 0 3px; margin:-13px 0; }
.row.done  .stripe { background:var(--ok); }
.row.part  .stripe { background:var(--warn); }
.row.block .stripe { background:var(--crit); }
.row .what { font-size:15px; }
.row .note { color:var(--muted); font-size:13.5px; }
.tag {
  font-family:var(--mono); font-size:11px; letter-spacing:.09em;
  text-transform:uppercase; padding:4px 9px; border-radius:999px; white-space:nowrap;
}
.done .tag  { color:var(--ok);   background:color-mix(in srgb, var(--ok) 13%, transparent); }
.part .tag  { color:var(--warn); background:color-mix(in srgb, var(--warn) 15%, transparent); }
.block .tag { color:var(--crit); background:color-mix(in srgb, var(--crit) 14%, transparent); }

.scroll { overflow-x:auto; margin-top:18px; border:1px solid var(--line); border-radius:4px; background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:14.5px; }
th,td { padding:11px 16px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }
th {
  font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); font-weight:600; background:var(--sunk);
}
tbody tr:last-child td { border-bottom:0; }
td.num { font-family:var(--mono); font-variant-numeric:tabular-nums; }
.big { font-weight:680; color:var(--accent); }
.bad { font-weight:680; color:var(--crit); }

pre {
  font-family:var(--mono); font-size:13px; line-height:1.72;
  background:var(--sunk); border:1px solid var(--line); border-radius:4px;
  padding:16px 18px; overflow-x:auto; margin:16px 0;
}
pre .c { color:var(--muted); }
code { font-family:var(--mono); font-size:.92em; background:var(--sunk); padding:1px 5px; border-radius:3px; }

figure { margin:22px 0 0; }
figure img { width:100%; height:auto; display:block; border:1px solid var(--line); border-radius:4px; }
figcaption { font-size:13px; color:var(--muted); margin-top:9px; }
.pair { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:20px; margin-top:22px; }
.pair figure { margin:0; }

.callout {
  border:1px solid var(--line); border-left:3px solid var(--crit);
  background:var(--surface); border-radius:3px; padding:18px 20px; margin-top:22px;
}
.callout h3 { margin-top:0; color:var(--crit); }

ol.steps { counter-reset:s; list-style:none; padding:0; margin:18px 0 0; display:flex; flex-direction:column; gap:14px; }
ol.steps li {
  counter-increment:s; position:relative; padding-left:42px;
  background:var(--surface); border:1px solid var(--line); border-radius:3px; padding:15px 18px 15px 46px;
}
ol.steps li::before {
  content:counter(s); position:absolute; left:16px; top:15px;
  font-family:var(--mono); font-size:12px; font-weight:660; color:var(--accent);
}
ol.steps b { font-weight:640; }
ol.steps .sub { color:var(--muted); font-size:14px; display:block; margin-top:3px; }

footer { margin-top:64px; padding-top:20px; border-top:1px solid var(--line); color:var(--muted); font-size:13.5px; }
:where(a,summary):focus-visible { outline:2px solid var(--accent); outline-offset:3px; border-radius:2px; }
@media (prefers-reduced-motion:reduce){ *{animation:none!important;transition:none!important} }
</style>

<div class="wrap">

<div class="eyebrow">Engineering capstone &nbsp;·&nbsp; sponsored by Johnson &amp; Johnson</div>
<h1>Warehouse Safety CV — progress report</h1>
<p class="lede prose">
  Computer vision that detects violations of J&amp;J's Life-Saving Rules from warehouse
  camera footage. This page summarises what is built, what is measured, and the one
  problem currently blocking the primary rule.
</p>
<div class="meta">
  <span>18 commits</span><span>91 tests passing</span><span>3 trained models</span>
  <span>Status: prototype</span>
</div>

<h2>What it actually does</h2>
<p class="prose">
  We do not train a model to recognise "unsafe behaviour" — no such dataset exists.
  We train one detector to find <b>people</b> and <b>forklifts</b>, download a pose
  model and a tracker, and express the rules as <b>geometry over tracked positions</b>.
  Distances are measured in floor metres via homography, never in pixels: two boxes
  200 px apart can be 1 m or 20 m apart depending on depth.
</p>

<h2>Status</h2>
<div class="board">
  <div class="row done"><span class="stripe"></span><span><span class="what">Detector — person + forklift</span><br><span class="note">mAP50:95 0.820 on real CCTV validation</span></span><span class="tag">Done</span></div>
  <div class="row done"><span class="stripe"></span><span><span class="what">Pipeline — video in, violation events out</span><br><span class="note">Annotated video + events.jsonl with evidence frames</span></span><span class="tag">Done</span></div>
  <div class="row done"><span class="stripe"></span><span><span class="what">Rule 3 — pedestrian too close to a working vehicle</span><br><span class="note">Precision 1.000, recall 1.000 across 178 clips</span></span><span class="tag">Done</span></div>
  <div class="row block"><span class="stripe"></span><span><span class="what">Rule 5 — driver's body outside the vehicle</span><br><span class="note">Logic correct and tested, but the detector cannot see seated drivers</span></span><span class="tag">Blocked</span></div>
  <div class="row part"><span class="stripe"></span><span><span class="what">Rules 4 &amp; 1 — walkways, phone use</span><br><span class="note">Implemented and unit-tested; not yet exercised on real footage</span></span><span class="tag">Partial</span></div>
  <div class="row done"><span class="stripe"></span><span><span class="what">Demo video &amp; write-up</span><br><span class="note">60-second annotated reel, full results document</span></span><span class="tag">Done</span></div>
</div>

<h2>Measured results</h2>
<div class="scroll">
<table>
  <thead><tr><th>What</th><th>Result</th><th>How it was measured</th></tr></thead>
  <tbody>
    <tr><td>Detector accuracy</td><td class="num big">mAP 0.820</td><td>1387 validation images, 667 real CCTV</td></tr>
    <tr><td>&nbsp;&nbsp;— forklift</td><td class="num">AP 0.830</td><td>same</td></tr>
    <tr><td>&nbsp;&nbsp;— person</td><td class="num">AP 0.777</td><td>same</td></tr>
    <tr><td>Rule 3 precision / recall</td><td class="num big">1.000 / 1.000</td><td>178 clips, scored per event</td></tr>
    <tr><td>Homography error</td><td class="num big">0.0001 m</td><td>vs exact simulator camera matrices</td></tr>
    <tr><td>Rule 5 logic scenarios</td><td class="num">7 / 7 pass</td><td>executable validation matrix</td></tr>
    <tr><td>Seated-driver detection</td><td class="num bad">0 / 16</td><td>clean images, driver plainly visible</td></tr>
  </tbody>
</table>
</div>
<p class="prose" style="font-size:14.5px;color:var(--muted)">
  Earlier synthetic-only models scored ~0.96, but their validation came from the same
  simulator as their training data. 0.820 against real CCTV is the first figure that
  reflects anything like deployment.
</p>

<h2>The pipeline, working</h2>
<figure>
  <img src="__DEMO__" alt="Four frames from the demo video showing forklifts and people detected with boxes, track IDs, and a red violation banner.">
  <figcaption>Frames from the 60-second demo reel. Orange = forklift, red = person, each with a persistent track ID. The red banner marks frames where Rule 3 fires. Distances are computed on the floor plane in metres, not in pixels.</figcaption>
</figure>

<h3>Which rules the footage can actually exercise</h3>
<p class="prose">
  The reel above shows Rule 3 only, and that is a property of the available footage
  rather than of the system. Rule 4 needs walkway markings to calibrate against;
  Rule 5 needs a driver in a cab; Rule 1 needs someone holding a phone. None of
  that exists in the clips we could obtain.
</p>
<div class="scroll">
<table>
  <thead><tr><th>Rule</th><th>Real footage</th><th>Generated scene</th></tr></thead>
  <tbody>
    <tr><td>Rule 3 — proximity</td><td class="num" style="color:var(--ok)">fires</td><td class="num" style="color:var(--ok)">fires</td></tr>
    <tr><td>Rule 4 — off walkway</td><td>no walkways marked</td><td class="num" style="color:var(--ok)">fires</td></tr>
    <tr><td>Driver association</td><td>no driver present</td><td class="num" style="color:var(--ok)">visible</td></tr>
    <tr><td>Rule 5 — body outside cab</td><td class="num bad">blocked</td><td>no cab in scene</td></tr>
    <tr><td>Rule 1 — phone use</td><td>no phone use filmed</td><td>—</td></tr>
  </tbody>
</table>
</div>
<figure>
  <img src="__RULES__" alt="Four frames of a generated warehouse scene: a yellow walkway outline, tracked people, and a banner reading VIOLATION rule(s): 3, 4.">
  <figcaption>The generated scene exercises two rules at once — the banner reads <b>[3, 4]</b>. The yellow outline is the walkway: the pedestrian inside it is never flagged, the one outside it is. <b>id1 P DRV</b> is the driver, correctly identified as riding the vehicle and excluded from Rule 3 — without that, every driver would read as a pedestrian standing 0 m from a moving forklift, and the system would alarm constantly. It looks schematic because every position in it is known exactly, which is what lets the events be checked against arithmetic instead of judged by eye.</figcaption>
</figure>

<h3>Every violation is an explainable record</h3>
<pre><span class="c">// one line of events.jsonl</span>
{ "rule": 3, "person_track": 0, "vehicle_track": 1,
  "distance_m": 2.21, "threshold_m": 8.1, "vehicle_speed_ms": 0.73,
  "start_s": 0.47, "end_s": 3.67, "duration_s": 3.2,
  "camera_id": "ceiling_00", "evidence_frame": "evt_00000_rule3.jpg" }</pre>
<p class="prose">
  A safety team can be told exactly why something was flagged — "a pedestrian came
  within 2.21 m of a moving forklift, threshold 8.1 m, for 3.2 seconds" — with the
  frame attached. That explainability is what makes the system adoptable.
</p>

<h2>Pose estimation works at camera distance</h2>
<figure>
  <img src="__POSE__" alt="Four warehouse frames with skeleton overlays correctly placed on workers.">
  <figcaption>RTMPose on workers 190–356 px tall: 17/17 keypoints on three of four, including a crouching worker. Rule 5 needs these joints to decide whether a driver has leaned outside the cab.</figcaption>
</figure>

<div class="callout">
  <h3>The blocker: the detector cannot see drivers</h3>
  <p>
    Our fine-tuned detector finds every forklift and <b>none of their drivers</b>. The
    COCO-pretrained model it started from could find some. Fine-tuning destroyed the
    capability.
  </p>
  <p>
    The cause is a labelling gap: only <b>4.9%</b> of forklift boxes in the training data
    contain a labelled person, while a driver is visibly present in most of them. An
    unlabelled object is trained as <i>background</i> — so the model learned that a seated
    driver is not a person.
  </p>
  <p style="margin-bottom:0">
    Rule 5 works by identifying the driver, then checking whether their keypoints leave
    the cab. With no driver detected, there is nothing to check.
  </p>
</div>

<div class="pair">
  <figure>
    <img src="__OURS__" alt="Forklifts with orange boxes; drivers clearly visible but none detected.">
    <figcaption><b>Our fine-tuned detector — 0 of 16.</b> Every forklift found, no driver found, though each cab holds a driver in a hard hat and hi-vis vest.</figcaption>
  </figure>
  <figure>
    <img src="__COCO__" alt="Same forklifts, with magenta boxes correctly marking five seated drivers.">
    <figcaption><b>COCO-pretrained, before fine-tuning — 5 of 16.</b> Magenta boxes are correct, with no false positives.</figcaption>
  </figure>
</div>

<h2>See it yourself in six seconds</h2>
<p class="prose">
  No footage, model weights or GPU needed. This builds a synthetic warehouse scene whose
  ground truth is exact arithmetic, runs the real pipeline over it, and checks the rules
  against it.
</p>
<pre>python -m scripts.make_synthetic_clip
python -m pytest -q                     <span class="c"># 91 tests, ~6 s</span>

<span class="c"># prove distances are real metres, not pixels</span>
python -m src.geometry data/calibration/synthetic_cam1.json \
       --pair 140 700 1140 700 24.0     <span class="c"># computed 24.00 m — PASS</span></pre>

<h2>What comes next</h2>
<ol class="steps">
  <li><b>Add the missing driver labels</b>
    <span class="sub">The one change that unblocks Rule 5. Roughly 800 images need a box drawn around the driver; automatic labelling recovers about a third, the rest is a few hours split across the team.</span></li>
  <li><b>Film the staged validation clips</b>
    <span class="sub">A driver leaning out of a cab appears in no dataset we could find — synthetic or real. One afternoon with a forklift and a safety supervisor produces all of it. The expected outcome of each clip is already written down as a test.</span></li>
  <li><b>Ask J&amp;J for a real forklift length and one camera's footage</b>
    <span class="sub">The 3-vehicle-length threshold currently uses a 2.7 m placeholder, and it scales every Rule 3 decision.</span></li>
</ol>

<footer>
  <p style="margin-top:0">
    All components are permissively licensed for commercial handoff: RF-DETR, RTMPose,
    ByteTrack and OpenCV under Apache-2.0 or MIT. Training data comes from NVIDIA
    PhysicalAI SDG-Warehouse (OpenMDW 1.1) and three Roboflow Universe datasets
    (CC BY 4.0, attribution required).
  </p>
  <p style="margin-bottom:0">
    No J&amp;J site footage has been used at any point — all results come from public and
    synthetic proxies, and may shift on real site data.
  </p>
</footer>

</div>
"""


def main():
    html = (HTML.replace('__DEMO__', img('demo'))
                .replace('__POSE__', img('pose'))
                .replace('__OURS__', img('ours'))
                .replace('__COCO__', img('coco'))
                .replace('__RULES__', img('rules')))
    out = os.path.join(HERE, 'progress.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'wrote {out}  ({os.path.getsize(out)/1024:.0f} KB)')


if __name__ == '__main__':
    main()
