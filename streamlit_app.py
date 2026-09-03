"""CARD-Net interactive demo — Streamlit Community Cloud edition.

Cascaded Adversarial-Robustness and Distribution-shift Defence Network.

Three views:
  1. Analyse      — run all five stages on an uploaded or sample image.
  2. Attack lab   — craft FGSM/PGD adversarial examples and compare the
                    undefended classifier against the full cascade.
  3. Results      — the Chapter 4 figures and tables, read from results/.

The 143 MB diffusion checkpoint is not kept in the repository. It is fetched
once, on first boot, from the public ddpm-torch release the experiments used,
and its byte count is checked; keeping it out of git avoids Git LFS entirely.
"""
import io
import os
import sys
import threading
import time
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(2)

CKPT_DIR = os.path.join(HERE, "checkpoints")
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")

DDPM_URL = ("https://github.com/tqch/ddpm-torch/releases/download/"
            "checkpoints/cifar10_2040.pt")
DDPM_PATH = os.path.join(CKPT_DIR, "ddpm_cifar10_2040.pt")
DDPM_BYTES = 143080283

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
ROUTE_COLOR = {"ACCEPT": "#2ecc71", "DEFER": "#f39c12", "FLAG": "#e74c3c"}

# The cascade is CPU-bound; serialise inferences so concurrent visitors queue
# rather than thrash the container's two cores.
LOCK = threading.Lock()

st.set_page_config(page_title="CARD-Net Demo", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _fetch_ddpm(progress):
    os.makedirs(CKPT_DIR, exist_ok=True)
    tmp = DDPM_PATH + ".part"
    with urllib.request.urlopen(DDPM_URL) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or DDPM_BYTES)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            progress.progress(min(done / total, 1.0),
                              text="Fetching the diffusion checkpoint — "
                                   f"{done/1e6:.0f} / {total/1e6:.0f} MB "
                                   "(first boot only)")
    if os.path.getsize(tmp) != DDPM_BYTES:
        os.remove(tmp)
        raise RuntimeError("Checkpoint download was truncated; reload to retry.")
    os.replace(tmp, DDPM_PATH)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load every stage once per container."""
    if not (os.path.exists(DDPM_PATH)
            and os.path.getsize(DDPM_PATH) == DDPM_BYTES):
        bar = st.progress(0.0, text="Fetching the diffusion checkpoint "
                                    "(first boot only)…")
        _fetch_ddpm(bar)
        bar.empty()

    with st.spinner("Loading the five-stage cascade…"):
        from cardnet import models
        models.CKPT = CKPT_DIR          # flat layout here, not the research tree
        from cardnet import purify, detect, fusion, router, attacks
        from cardnet.certify import SmoothedClassifier

        cal = torch.load(os.path.join(CKPT_DIR, "calibration.pt"),
                         map_location="cpu", weights_only=False)

        # `enc` is the undefended classifier: the attack target and the
        # undefended baseline. Stage 2 scores on its own hardened backbone.
        enc = models.load_pretrained("resnet20")
        hard = os.path.join(CKPT_DIR, "hardened_encoder_resnet20.pt")
        det_enc = (models.load_local(hard, "resnet20")
                   if os.path.exists(hard) else enc)

        unet, dcfg = models.load_unet()
        pur = purify.AdaptivePurifier(unet, dcfg, t_max=cal["t_max"],
                                      stride=cal["stride"], gamma=cal["gamma"],
                                      tau=cal["tau"])
        det = detect.ShiftDetector(det_enc).load_state(cal["detector"])
        fusion.MODE = cal.get("fusion_mode", "linear")

        sp = os.path.join(CKPT_DIR, "student_resnet20.pt")
        student = (models.load_local(sp) if os.path.exists(sp)
                   else models.load_pretrained("resnet20"))
        smoother = SmoothedClassifier(student, sigma=0.25, n=32)

        pp = os.path.join(CKPT_DIR, "pgd_at_resnet32.pt")
        pgd_at = models.load_local(pp, arch="resnet32") if os.path.exists(pp) else None

    return dict(cal=cal, enc=enc, pur=pur, det=det, smoother=smoother,
                pgd_at=pgd_at, fusion=fusion, router=router, attacks=attacks,
                kappa=float(cal.get("fusion_kappa", 1.0)),
                hardened=os.path.exists(hard))


@st.cache_data(show_spinner=False)
def load_samples():
    d = np.load(os.path.join(HERE, "data", "cifar10_demo.npz"))
    return d["x"], d["y"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def to_tensor(arr_uint8):
    t = torch.from_numpy(np.ascontiguousarray(arr_uint8)).float().div_(255.)
    return t.permute(0, 3, 1, 2).contiguous()


def to_pil(t, size=256):
    a = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(a).resize((size, size), Image.NEAREST)


@torch.no_grad()
def run_cascade(P, x):
    t0 = time.time()
    t = time.time()
    s_pre = P["det"].pre_severity(x)
    depth = int(P["pur"].depth(s_pre).item())
    x_hat = P["pur"].purify(x, s_pre=s_pre)
    t1 = time.time() - t

    t = time.time()
    sev = P["det"].severity(x, x_hat)
    s, s_pert, s_nov = sev[0], sev[1], sev[2]
    t2 = time.time() - t

    t = time.time()
    p = P["smoother"].soft_probs(x_hat)
    t3 = time.time() - t

    t = time.time()
    pt = P["fusion"].fuse(p, s, kappa=P["kappa"])
    u, H = P["fusion"].uncertainty(pt)
    t4 = time.time() - t

    r = int(P["router"].route(u, P["cal"]["tau_acc"], P["cal"]["tau_def"]).item())
    probs = pt.squeeze(0).numpy()
    order = np.argsort(probs)[::-1][:5]
    return dict(
        pred=CLASSES[int(pt.argmax(1).item())],
        conf=float(pt.max(1).values.item()) * 100,
        top5=[(CLASSES[i], float(probs[i]) * 100) for i in order],
        route=["ACCEPT", "DEFER", "FLAG"][r],
        depth=depth, severity=float(s.item()),
        s_pert=float(s_pert.item()), s_nov=float(s_nov.item()),
        s_pre=float(s_pre.item()), u=float(u.item()), H=float(H.item()),
        smooth_conf=float(p.max(1).values.item()) * 100,
        x_hat=x_hat, total=(time.time() - t0) * 1000,
        stages=[("Stage 1 — purification", t1 * 1000),
                ("Stage 2 — detection", t2 * 1000),
                ("Stage 3 — smoothing", t3 * 1000),
                ("Stage 4 — fusion", t4 * 1000)])


@torch.no_grad()
def run_plain(model, x):
    t0 = time.time()
    p = torch.softmax(model(x), dim=1)
    return dict(pred=CLASSES[int(p.argmax(1).item())],
                conf=float(p.max(1).values.item()) * 100,
                ms=(time.time() - t0) * 1000)


def verdict_card(col, title, res, truth):
    ok = truth is not None and res["pred"] == truth
    colour = "#2ecc71" if ok else ("#e74c3c" if truth else "#94a3b8")
    tail = ""
    if "route" in res:
        tail = (f"<div style='font-size:.75rem;color:#94a3b8;margin-top:4px'>"
                f"router: {res['route']} · s(x)={res['severity']:.3f}</div>")
    col.markdown(
        f"<div style='background:#161829;border:1px solid #2a2d3e;"
        f"border-radius:10px;padding:14px;text-align:center'>"
        f"<div style='font-size:.7rem;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:.8px'>{title}</div>"
        f"<div style='font-size:1.35rem;font-weight:700;color:{colour};"
        f"margin:6px 0 2px'>{res['pred'].capitalize()}</div>"
        f"<div style='font-size:.8rem;color:#94a3b8'>{res['conf']:.1f}%"
        + (f" · {'correct' if ok else 'wrong'}" if truth else "")
        + f"</div>{tail}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def view_analyse(P, X, Y):
    st.subheader("Run the cascade on an image")
    c1, c2 = st.columns([1, 1])
    with c1:
        up = st.file_uploader("Upload any image (resized to 32×32)",
                              type=["png", "jpg", "jpeg", "bmp", "webp"])
    with c2:
        i = st.selectbox(
            "…or pick a CIFAR-10 test image",
            options=list(range(24)),
            format_func=lambda k: f"#{k} — {CLASSES[int(Y[k])]}")
        st.image(Image.fromarray(X[i]).resize((96, 96), Image.NEAREST))

    if not st.button("▶ Run CARD-Net", type="primary"):
        return

    if up is not None:
        img = Image.open(io.BytesIO(up.read())).convert("RGB") \
                   .resize((32, 32), Image.BILINEAR)
        x = to_tensor(np.asarray(img, dtype=np.uint8)[np.newaxis])
        truth = None
    else:
        x = to_tensor(X[i][np.newaxis])
        truth = CLASSES[int(Y[i])]

    with st.spinner("Running stages 1–5… a few seconds on CPU."):
        with LOCK:
            r = run_cascade(P, x)
            base = run_plain(P["enc"], x)

    a, b = st.columns(2)
    a.image(to_pil(x), caption="Input")
    b.image(to_pil(r["x_hat"]), caption=f"Purified · T = {r['depth']} steps")

    st.markdown(
        f"### {r['pred'].capitalize()} &nbsp;<span style='font-size:1rem;"
        f"color:#94a3b8'>{r['conf']:.1f}% confidence</span> &nbsp;"
        f"<span style='background:{ROUTE_COLOR[r['route']]};color:#fff;"
        f"padding:3px 12px;border-radius:14px;font-size:.8rem'>"
        f"{r['route']}</span>", unsafe_allow_html=True)
    st.caption(
        (f"Ground truth: **{truth}** · " if truth else "")
        + f"undefended ResNet-20 said **{base['pred']}** ({base['conf']:.1f}%)")

    m = st.columns(5)
    for col, (lab, val) in zip(m, [
            ("Stage 1 · depth T(x)", r["depth"]),
            ("Stage 2 · severity", f"{r['severity']:.3f}"),
            ("Stage 3 · smoothed", f"{r['smooth_conf']:.1f}%"),
            ("Stage 4 · uncertainty", f"{r['u']:.3f}"),
            ("Stage 5 · route", r["route"])]):
        col.metric(lab, val)

    l, rr = st.columns(2)
    with l:
        st.markdown("**Top-5 posterior**")
        st.bar_chart(pd.DataFrame({"%": [v for _, v in r["top5"]]},
                                  index=[k for k, _ in r["top5"]]))
    with rr:
        st.markdown("**Stage 2 — the two channels**")
        st.dataframe(pd.DataFrame({
            "value": [f"{r['s_pert']:.4f}", f"{r['s_nov']:.4f}",
                      f"{r['severity']:.4f}", f"{r['s_pre']:.4f}",
                      f"{r['H']:.4f}"]},
            index=["Perturbation score (adversarial channel)",
                   "Novelty score (OOD channel)",
                   "Severity = max(perturbation, novelty)",
                   "Pre-estimate s₀(x) → depth",
                   "Entropy H(x)"]), use_container_width=True)

    st.markdown("**Latency**")
    st.dataframe(pd.DataFrame({"ms": [f"{v:.0f}" for _, v in r["stages"]]},
                              index=[k for k, _ in r["stages"]]),
                 use_container_width=True)
    st.caption(f"Total {r['total']/1000:.2f} s · undefended forward pass "
               f"{base['ms']:.1f} ms")


def view_attack(P, X, Y):
    st.subheader("Craft an adversarial example")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    i = c1.selectbox("CIFAR-10 test image", options=list(range(24)),
                     format_func=lambda k: f"#{k} — {CLASSES[int(Y[k])]}",
                     key="atk_idx")
    kind = c2.radio("Attack", ["PGD", "FGSM"], horizontal=True)
    eps = c3.slider("Budget ε (/255)", 1, 16, 8)
    steps = c3.slider("PGD steps", 1, 20, 10) if kind == "PGD" else 1

    st.caption("The attack is white-box against the **undefended** ResNet-20 and "
               "is then presented to each defence — the grey-box threat model of "
               "Chapter 3. It is not an adaptive (BPDA) attack on the cascade "
               "itself; that result is in the Results tab.")

    if not st.button("▶ Attack and defend", type="primary"):
        return

    x = to_tensor(X[i][np.newaxis])
    y = int(Y[i])
    truth = CLASSES[y]
    with st.spinner("Crafting the attack and running the cascade twice…"):
        with LOCK:
            yt = torch.tensor([y])
            if kind == "FGSM":
                xa = P["attacks"].fgsm(P["enc"], x, yt, eps / 255.0)
            else:
                xa = P["attacks"].pgd(P["enc"], x, yt, eps / 255.0, steps=steps)
            clean = {"undefended": run_plain(P["enc"], x)}
            adv = {"undefended": run_plain(P["enc"], xa)}
            if P["pgd_at"] is not None:
                clean["pgd_at"] = run_plain(P["pgd_at"], x)
                adv["pgd_at"] = run_plain(P["pgd_at"], xa)
            clean["cardnet"] = run_cascade(P, x)
            adv["cardnet"] = run_cascade(P, xa)

    d = (xa - x).abs()
    a, b, c = st.columns(3)
    a.image(to_pil(x), caption="Clean input")
    b.image(to_pil(d / max(d.max().item(), 1e-8)),
            caption="Perturbation (contrast-normalised)")
    c.image(to_pil(xa), caption=f"{kind} adversarial (ε = {eps}/255)")
    st.caption(f"Ground truth **{truth}** · measured ‖δ‖∞ = "
               f"{d.max().item()*255:.1f}/255")

    names = {"undefended": "Undefended ResNet-20",
             "pgd_at": "PGD-AT ResNet-32", "cardnet": "CARD-Net (5 stages)"}
    st.markdown("#### On the clean image")
    cols = st.columns(len(clean))
    for col, k in zip(cols, ["undefended", "pgd_at", "cardnet"]):
        if k in clean:
            verdict_card(col, names[k], clean[k], truth)
    st.markdown("#### On the adversarial image")
    cols = st.columns(len(adv))
    for col, k in zip(cols, ["undefended", "pgd_at", "cardnet"]):
        if k in adv:
            verdict_card(col, names[k], adv[k], truth)
    st.caption("Green marks a prediction still matching the ground-truth label. "
               "A single image is anecdotal — the aggregate accuracies over "
               "240 images × 3 seeds are in the Results tab.")


FIGURES = [
    ("fig4_05_robust_accuracy.png", "Clean and robust accuracy across the attack suite, mean ± sd over three seeds."),
    ("fig4_07_ood_auroc.png", "Out-of-distribution detection AUROC on one near-OOD and two far-OOD sets."),
    ("fig4_08_adv_detection.png", "Adversarial-input detection: separating attacked inputs from clean ones."),
    ("fig4_09_corruption.png", "Corruption robustness averaged over corruption types, by severity."),
    ("fig4_09b_corruption_per_type.png", "Per-corruption accuracy at the highest severity."),
    ("fig4_10_ablation.png", "Stage ablation: accuracy change when one stage is removed."),
    ("fig4_11_latency_routing.png", "Inference cost per image, and how the router splits traffic."),
    ("fig4_12_bpda.png", "Adaptive adversary — BPDA straight through Stage 1."),
    ("fig4_15_certified.png", "Certified accuracy against ℓ₂ radius under randomised smoothing."),
    ("fig4_13_continual.png", "One continual-adaptation cycle under covariate and concept drift."),
    ("fig4_14_significance.png", "Paired significance tests with 95% confidence intervals."),
    ("fig4_16_qualitative_purification.png", "What Stage 1 does to clean and PGD-attacked inputs."),
]

LBL = {"cardnet": "CARD-Net (full)", "undefended": "Undefended ResNet-20",
       "undefended_vgg": "Undefended VGG-16-BN", "pgd_at": "PGD-AT ResNet-32",
       "trades": "TRADES ResNet-20", "rand_smooth": "Randomised smoothing",
       "abl_no_purify": "− Stage 1", "abl_no_detect": "− Stage 2",
       "abl_no_smooth": "− Stage 3", "abl_no_fusion": "− Stage 4",
       "abl_single_teacher": "Single teacher"}
ORDER = ["cardnet", "undefended", "undefended_vgg", "pgd_at", "trades",
         "rand_smooth"]


def view_results():
    st.subheader("Experimental results")
    st.caption("Every figure and table below is generated from the same stored "
               "result files, so the charts and the numbers cannot drift apart.")

    def read(name):
        p = os.path.join(RES_DIR, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    s, o, ad = read("summary.csv"), read("ood_metrics.csv"), \
        read("adv_detection_metrics.csv")

    if s is not None:
        adv = s[s.part == "adv"]
        conds = [c for c in ["clean", "fgsm_8", "pgd10_8", "pgd40_8",
                             "autoattack_8", "cw"] if c in set(adv.condition)]
        rows = []
        for m in ORDER:
            sub = adv[adv.model == m].set_index("condition")
            if sub.empty:
                continue
            rows.append([LBL.get(m, m)] + [
                f"{100*sub.acc_mean[c]:.1f}" if c in sub.index else "—"
                for c in conds])
        st.markdown("**Robust accuracy (%) under L∞ attack**")
        st.dataframe(pd.DataFrame(rows, columns=["Defence"] + conds)
                     .set_index("Defence"), use_container_width=True)

    if o is not None:
        g = o.groupby(["model", "ood_set"]).auroc.mean() * 100
        sets = ["near_cifar100", "far_mnist", "far_fmnist"]
        rows = [[LBL.get(m, m)] + [f"{g[(m, x)]:.1f}" if (m, x) in g.index else "—"
                                   for x in sets]
                for m in ORDER if any((m, x) in g.index for x in sets)]
        st.markdown("**Out-of-distribution detection (AUROC ×100)**")
        st.dataframe(pd.DataFrame(rows, columns=["Defence"] + sets)
                     .set_index("Defence"), use_container_width=True)

    if ad is not None:
        g = ad.groupby(["model", "condition"]).auroc.mean() * 100
        cs = ["fgsm_8", "pgd10_8", "pgd40_8", "autoattack_8", "cw"]
        rows = [[LBL.get(m, m)] + [f"{g[(m, c)]:.1f}" if (m, c) in g.index else "—"
                                   for c in cs]
                for m in ORDER if any((m, c) in g.index for c in cs)]
        st.markdown("**Adversarial-input detection (AUROC ×100)**")
        st.dataframe(pd.DataFrame(rows, columns=["Defence"] + cs)
                     .set_index("Defence"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Figures")
    for j in range(0, len(FIGURES), 2):
        cols = st.columns(2)
        for col, (fn, cap) in zip(cols, FIGURES[j:j + 2]):
            p = os.path.join(FIG_DIR, fn)
            if os.path.exists(p):
                col.image(p, caption=cap, use_container_width=True)


def view_about(P):
    st.subheader("What this is")
    st.write(
        "CARD-Net is a five-stage inference-time defence cascade for image "
        "classifiers, evaluated on CIFAR-10. Every number in the Results tab "
        "was produced by the code in this repository; nothing is illustrative.")
    cols = st.columns(5)
    for col, (n, t) in zip(cols, [
            ("Stage 1", "Adaptive diffusion purification — input-dependent depth T(x) via strided DDIM"),
            ("Stage 2", "Dual-channel shift detection — purification residual, and free energy + Mahalanobis novelty"),
            ("Stage 3", "Randomised smoothing over a distilled student"),
            ("Stage 4", "Reliability-weighted fusion of the posterior"),
            ("Stage 5", "Router — accept, defer or flag on calibrated uncertainty")]):
        col.markdown(f"**{n}**  \n<span style='font-size:.8rem;color:#94a3b8'>"
                     f"{t}</span>", unsafe_allow_html=True)

    st.markdown("### What to expect — and what the trade-off costs")
    st.write(
        "The defence is **not** free, and this demo does not hide the price. On "
        "clean images CARD-Net reaches **60.0%** accuracy against the "
        "undefended network's **91.2%**, so roughly two clean images in five "
        "come back wrong — usually with the router marking them **DEFER**, "
        "which is the cascade correctly reporting its own low confidence rather "
        "than a bug. Confidences read low for the same reason: Stage 4 "
        "deliberately moves probability mass away from the classifier's opinion "
        "in proportion to the measured severity.")
    st.write(
        "What it buys is robustness. Under PGD-10 at ε = 8/255 the undefended "
        "network drops to **0.0%** while CARD-Net holds **38.2%**, above the "
        "adversarially-trained baseline's 33.1%. On out-of-distribution "
        "detection it beats every defended baseline by 29–47 AUROC points and "
        "reaches parity with an undefended maximum-softmax baseline.")
    st.caption(
        f"Stage 2 backbone: {'outlier-exposure hardened' if P['hardened'] else 'standard'}"
        " · Running on CPU with two threads; one image takes a few seconds, "
        "almost all of it Stage 1. Requests are served one at a time.")


# ---------------------------------------------------------------------------
def main():
    st.title("⚡ CARD-Net")
    st.caption("Cascaded Adversarial-Robustness and Distribution-shift Defence "
               "Network — five-stage inference cascade on CIFAR-10")

    P = load_pipeline()
    X, Y = load_samples()

    t1, t2, t3, t4 = st.tabs(["Analyse an image", "Attack lab",
                              "Experimental results", "About"])
    with t1:
        view_analyse(P, X, Y)
    with t2:
        view_attack(P, X, Y)
    with t3:
        view_results()
    with t4:
        view_about(P)


if __name__ == "__main__":
    main()
