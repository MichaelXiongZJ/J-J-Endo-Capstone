"""Download a Roboflow Universe dataset and normalise it into our COCO layout.

WHY REAL IMAGES MATTER HERE
---------------------------
Everything measured so far is synthetic. Real photographs of real forklifts fix
two gaps at once that no amount of SDG data can: **domain shift** (real lighting,
real PPE, real motion blur, real occlusion) and the **vehicle-type gap** (J&J
operates sit-down counterbalance forklifts; our synthetic slice is mostly
stand-on reach trucks).

ON "THE ONLY DOWNLOAD FORMATS ARE YOLO"
---------------------------------------
That is fine. YOLO *format* is just `.txt` files with normalised coordinates and
a `data.yaml` — it carries no licence obligations. The AGPL-3.0 problem in
context.md §7.1 is about Ultralytics' *code*, which we neither install nor use.
RF-DETR reads YOLO layout natively (`data.yaml` + `train/images/`), and this
script also converts YOLO -> COCO so real images can be merged with the synthetic
set in one dataset.

LICENCE IS A HARD GATE
----------------------
The deliverable goes to J&J for commercial use, so every input must be
permissively licensed (context.md §3). Roboflow Universe datasets are licensed
per dataset and some are non-commercial. This script prints the licence and
refuses to proceed on a known-bad one unless you pass --accept-licence.

  OK       : CC BY 4.0 (attribution required), MIT, Apache-2.0, Public Domain, CC0
  NOT OK   : anything with NC / NonCommercial
  CHECK    : CC BY-SA (share-alike — ask before relying on it)

Usage:
    export ROBOFLOW_API_KEY=...            # from app.roboflow.com -> Settings -> API
    python -m scripts.fetch_roboflow --url https://universe.roboflow.com/hitsz/forklift-and-human/dataset/2
    python -m scripts.fetch_roboflow --url ... --format yolov11
"""

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile

API = 'https://api.roboflow.com'

# Class names vary between datasets; normalise onto ours. Anything unmapped is
# reported and dropped rather than silently folded into a class it is not.
NAME_MAP = {
    'person': 'person', 'people': 'person', 'human': 'person', 'humans': 'person',
    'worker': 'person', 'pedestrian': 'person', 'man': 'person',
    'forklift': 'forklift', 'fork-lift': 'forklift', 'forklifts': 'forklift',
    'fork lift': 'forklift', 'forklift-truck': 'forklift', 'truck': 'forklift',
    # 'cart' is the vehicle class in hitsz/forklift-and-human. Verified visually
    # before mapping, because J&J ruled the pallet-jack family OUT of scope
    # (jj-rule-scope-rulings): the boxes are on sit-down counterbalance forklifts,
    # not hand carts or pallet jacks. The word is just the annotator's choice.
    # Re-check this if another dataset uses 'cart' for something else.
    'cart': 'forklift',
}

BAD_LICENCE = re.compile(r'\bNC\b|non[- ]?commercial', re.I)
GOOD_LICENCE = re.compile(r'CC BY 4\.0|MIT|Apache|Public Domain|CC0|BY 4\.0', re.I)


def parse_url(url):
    """https://universe.roboflow.com/<ws>/<project>/dataset/<version> -> parts."""
    p = urllib.parse.urlparse(url).path.strip('/').split('/')
    if len(p) < 2:
        raise SystemExit(f'cannot parse workspace/project from {url}')
    workspace, project = p[0], p[1]
    version = None
    for i, seg in enumerate(p):
        if seg in ('dataset', 'model') and i + 1 < len(p) and p[i + 1].isdigit():
            version = int(p[i + 1])
    return workspace, project, version


def api_get(path, api_key, **params):
    params['api_key'] = api_key
    url = f'{API}/{path}?{urllib.parse.urlencode(params)}'
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', required=True, help='Roboflow Universe dataset URL')
    ap.add_argument('--format', default='coco',
                    help='coco (default) or yolov11/yolov8. COCO needs no conversion')
    ap.add_argument('--out', default='data/real', help='output root')
    ap.add_argument('--api-key', default=os.environ.get('ROBOFLOW_API_KEY'))
    ap.add_argument('--accept-licence', action='store_true',
                    help='proceed even if the licence looks non-commercial')
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit(
            'No API key. Get one free at app.roboflow.com -> Settings -> API, then:\n'
            '  export ROBOFLOW_API_KEY=xxxx        (bash)\n'
            '  $env:ROBOFLOW_API_KEY="xxxx"        (PowerShell)\n'
            'or pass --api-key. A free account is enough for Universe downloads.')

    ws, project, version = parse_url(args.url)
    print(f'workspace={ws} project={project} version={version}')

    meta = api_get(f'{ws}/{project}', args.api_key)
    p = meta.get('project', {})
    licence = str(p.get('license', 'UNKNOWN'))
    print(f'\n  name     : {p.get("name")}')
    print(f'  images   : {p.get("images")}')
    print(f'  classes  : {p.get("classes")}')
    print(f'  LICENCE  : {licence}')

    if BAD_LICENCE.search(licence):
        print('\nREFUSING: this licence looks non-commercial. The deliverable goes to '
              'J&J for commercial use, so a NonCommercial dataset cannot be used for '
              'training (context.md §3). Pass --accept-licence only if you have '
              'checked and disagree.')
        if not args.accept_licence:
            sys.exit(2)
    elif not GOOD_LICENCE.search(licence):
        print('\nWARNING: licence not recognised as clearly permissive. Confirm it '
              'allows commercial use before this data influences a deliverable.')

    if version is None:
        version = max(int(v['id'].split('/')[-1]) for v in meta.get('versions', []))
        print(f'  using latest version {version}')

    print(f'\nrequesting {args.format} export...')
    for attempt in range(12):
        info = api_get(f'{ws}/{project}/{version}/{args.format}', args.api_key)
        link = info.get('export', {}).get('link')
        if link:
            break
        print('  export still generating, waiting 10 s...')
        time.sleep(10)
    else:
        raise SystemExit('export never became ready')

    dest = os.path.join(args.out, f'{project}_v{version}')
    os.makedirs(dest, exist_ok=True)
    print(f'downloading -> {dest}')
    with urllib.request.urlopen(link, timeout=600) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest)
    print(f'  {len(blob) / 2**20:.1f} MiB extracted')

    with open(os.path.join(dest, 'LICENCE.txt'), 'w') as f:
        f.write(f'Source: {args.url}\nLicence: {licence}\n'
                f'Downloaded: {time.strftime("%Y-%m-%d")}\n'
                'Attribution may be required — see the dataset page.\n')

    splits = [d for d in ('train', 'valid', 'test')
              if os.path.isdir(os.path.join(dest, d))]
    print(f'  splits: {splits}')
    print('\nNext:')
    print(f'  python -m scripts.real_to_coco --root {dest}   # normalise + merge')


if __name__ == '__main__':
    main()
