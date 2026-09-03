# CARD-Net — interactive demo

**Cascaded Adversarial-Robustness and Distribution-shift Defence Network.**
A five-stage inference-time defence cascade for image classifiers, evaluated on
CIFAR-10. This app runs the real pipeline on the real checkpoints — nothing is
simulated.

## Deploy it free on Streamlit Community Cloud

Community Cloud is free, needs only a GitHub account (no card), and gives an app
up to 2.7 GB of memory — comfortably more than this needs.

1. **Push this folder to a public GitHub repository.**

   ```bash
   cd cardnet_streamlit
   git init
   git add -A
   git commit -m "CARD-Net interactive demo"
   git branch -M main
   git remote add origin https://github.com/YOURNAME/cardnet-demo.git
   git push -u origin main
   ```

   The 143 MB diffusion checkpoint is deliberately **not** committed — it is in
   `.gitignore`, and the app downloads it on first boot from its public release.
   That keeps the repo about 18 MB and avoids Git LFS entirely.

2. **Deploy.** Go to <https://share.streamlit.io>, sign in with GitHub, click
   *New app*, choose your repository, branch `main`, main file
   `streamlit_app.py`, and click *Deploy*.

3. The first boot takes a few minutes: it installs PyTorch, then downloads the
   checkpoint with a progress bar. After that the app is live at
   `https://YOURNAME-cardnet-demo-....streamlit.app` — that URL is the link to
   share.

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What you can do

- **Analyse an image** — upload anything, or pick a CIFAR-10 test image, and
  watch all five stages run with a per-stage trace and latency breakdown.
- **Attack lab** — craft an FGSM or PGD adversarial example at a chosen ε and
  compare the undefended ResNet-20, a PGD-adversarially-trained ResNet-32, and
  the full cascade on the same input.
- **Experimental results** — the Chapter 4 tables and all twelve figures, read
  directly from the stored `results/` files.

## What to expect

The defence is not free. On clean images CARD-Net reaches **60.0%** accuracy
against the undefended network's **91.2%**, and predicted confidences are low by
construction, because Stage 4 moves probability mass away from the classifier in
proportion to measured severity. What it buys is robustness: under PGD-10 at
ε = 8/255 the undefended network falls to **0.0%** while CARD-Net holds
**38.2%**, above the PGD-adversarially-trained baseline's 33.1%. On
out-of-distribution detection it leads every defended baseline by 29–47 AUROC
points and reaches parity with an undefended maximum-softmax baseline.

## Notes on the free tier

- Apps hibernate after 12 hours of inactivity and wake on the next visit; the
  first request after waking re-downloads nothing but does reload the model, so
  give it 30–60 seconds. Open the link yourself shortly before sharing it.
- Inference is serialised behind a lock: concurrent visitors queue rather than
  compete for the container's cores.

## Threat model in the attack lab

Adversarial examples are crafted **white-box against the undefended classifier**
and then presented to each defence — the grey-box setting used for the main
results tables. It is *not* an adaptive attack on the cascade itself; the BPDA
adaptive-attack result is in the Results tab.

## Layout

```
streamlit_app.py     the app
cardnet/             the five stages: purify, detect, certify, fusion, router
ddpm_torch/          UNet definition for the diffusion purifier
pytorch_cifar_models/ ResNet/VGG backbone definitions
configs/cifar10.json DDPM schedule
checkpoints/         encoder, hardened detector backbone, distilled student,
                     PGD-AT baseline, and the fitted calibration object
                     (the DDPM checkpoint is fetched at runtime)
data/                a 400-image stratified CIFAR-10 test subset
figures/             the twelve Chapter 4 figures
results/             stored experimental results rendered by the Results tab
```
