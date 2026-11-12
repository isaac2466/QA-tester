"""Entrena el detector de elementos (cabeza sobre DINOv2) con pantallas sintéticas.

    cd backend
    python -m app.ai.train_detector --train 300 --val 40 --epochs 40

Guarda:  app/ai/weights/detector.pt  y  app/ai/weights/detector_metrics.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image

from app.ai.dataset import make_page
from app.ai.detector import CLASSES, DetectorHead, decode, dino_grid, loss_fn, targets
from app.services import dino as dino_service

ROOT = Path(__file__).resolve().parents[2]  # backend/
DATA = ROOT / "ai_data"
WEIGHTS = Path(__file__).parent / "weights"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


def build_split(name, n, seed, dino):
    """Genera las pantallas y guarda sus características de DINOv2 (se calculan una sola vez)."""
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    items = []
    t0 = time.time()
    for i in range(n):
        png, js, pt = d / f"pantalla_{i:04d}.png", d / f"pantalla_{i:04d}.json", d / f"pantalla_{i:04d}.dino.pt"
        if not png.exists():
            img, meta = make_page(seed * 100000 + i)
            img.save(png)
            js.write_text(json.dumps(meta), encoding="utf-8")
        meta = json.loads(js.read_text(encoding="utf-8"))
        if not pt.exists():
            torch.save(dino_grid(dino, Image.open(png).convert("RGB")).half(), pt)
        items.append((pt, meta))
        if (i + 1) % 20 == 0:
            print(f"  {name}: {i + 1}/{n} pantallas listas ({time.time() - t0:.0f}s)", flush=True)
    return items


def evaluate(head, items, thr=0.3):
    head.eval()
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
    with torch.inference_mode():
        for pt, meta in items:
            f = torch.load(pt).float()
            W, H = meta["size"]
            dets = decode(head(f[None]), W, H, thr=thr)
            for c in CLASSES:
                gt = [e["box"] for e in meta["elements"] if ("texto" if e["type"] == "texto" else "bloque") == c]
                pr = [d["box"] for d in dets if d["kind"] == c]
                used = set()
                for p in pr:
                    j = max(range(len(gt)), key=lambda k: iou(p, gt[k]), default=None)
                    if j is not None and j not in used and iou(p, gt[j]) >= 0.5:
                        used.add(j)
                        stats[c]["tp"] += 1
                    else:
                        stats[c]["fp"] += 1
                stats[c]["fn"] += len(gt) - len(used)
    out = {}
    for c, s in stats.items():
        out[c] = {"recall": round(s["tp"] / max(1, s["tp"] + s["fn"]), 3),
                  "precision": round(s["tp"] / max(1, s["tp"] + s["fp"]), 3), **s}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--resume", action="store_true", help="seguir entrenando el detector actual en vez de empezar de cero")
    a = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    print("Cargando DINOv2...", flush=True)
    dino_service.load()
    dino = dino_service.get()
    print("Preparando datos (generar pantallas + pasarlas por DINOv2):", flush=True)
    train = build_split("train", a.train, seed=1, dino=dino)
    val = build_split("val", a.val, seed=2, dino=dino)

    tg = {}
    for pt, meta in train + val:
        f = torch.load(pt)
        tg[pt] = targets(meta["elements"], *meta["size"], f.shape[1], f.shape[2])

    head = DetectorHead()
    if a.resume and (WEIGHTS / "detector.pt").exists():
        head.load_state_dict(torch.load(WEIGHTS / "detector.pt"))
        print("Continuando desde el detector actual", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    print(f"Entrenando {a.epochs} épocas con {len(train)} pantallas...", flush=True)
    for ep in range(a.epochs):
        head.train()
        random.shuffle(train)
        total = 0.0
        for pt, _ in train:
            f = torch.load(pt).float()
            loss = loss_fn(head(f[None]), [t[None] for t in tg[pt]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        msg = f"  época {ep + 1:2d}/{a.epochs}  pérdida {total / len(train):.3f}"
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            m = evaluate(head, val)
            msg += "  | val: " + "  ".join(f"{c} recall {m[c]['recall']:.2f} precisión {m[c]['precision']:.2f}" for c in CLASSES)
        print(msg, flush=True)

    WEIGHTS.mkdir(exist_ok=True)
    torch.save(head.state_dict(), WEIGHTS / "detector.pt")
    metrics = {"train_pantallas": len(train), "val_pantallas": len(val), "epocas": a.epochs, "val": evaluate(head, val)}
    (WEIGHTS / "detector_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Listo:", json.dumps(metrics["val"], ensure_ascii=False))


if __name__ == "__main__":
    main()
