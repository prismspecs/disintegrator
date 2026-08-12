"""Pack the run's frames and metrics into a single JSON blob for the write-up.

Frames become base64 data URIs because the Artifact CSP blocks external hosts:
a small set for the step scrubber, larger plates for the figure comparisons.
"""

import base64, csv, io, json, os, sys

from PIL import Image

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
DISTS = ["uniform", "gaussian"]
PLATE_STEPS = [0, 4, 8, 12, 16, 20]

SCRUB = dict(size=340, quality=70)   # every step, driven by the slider
PLATE = dict(size=460, quality=80)   # key steps, shown large


def encode(path, size, quality):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    out = {"dists": {}, "plateSteps": PLATE_STEPS}
    out["meta"] = json.load(open(f"{RESULTS}/meta.json"))

    total = 0
    for dist in DISTS:
        rows = list(csv.DictReader(open(f"{RESULTS}/{dist}.csv")))
        for r in rows:
            for k, v in r.items():
                r[k] = int(v) if k in ("step", "jpg_bytes") else float(v)

        frames, plates = [], {}
        for r in rows:
            p = f"{RESULTS}/{dist}/frame_{r['step']:02d}.jpg"
            frames.append(encode(p, **SCRUB))
            if r["step"] in PLATE_STEPS:
                plates[str(r["step"])] = encode(p, **PLATE)
        out["dists"][dist] = {"rows": rows, "frames": frames, "plates": plates}
        total += sum(len(f) for f in frames) + sum(len(p) for p in plates.values())
        print(f"{dist}: {len(rows)} rows, {len(frames)} frames, {len(plates)} plates")

    json.dump(out, open("assets.json", "w"))
    print(f"assets.json: {os.path.getsize('assets.json')/1e6:.2f} MB "
          f"(image payload {total/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
