"""
Combined Inference: Classification + Segmentation with Grad-CAM
================================================================
Runs both the classification student (ConvNeXt-Tiny) and segmentation
student (SegFormer-B2) on the same X-ray images, producing a single
multi-panel visualisation per sample that includes:

  Row 1: Original | Ground-Truth Mask | Predicted Mask | Overlay (pred on orig)
  Row 2: Seg Grad-CAM | Cls Grad-CAM | Classification Bar-Chart | Summary Card

Results and per-sample PNGs are stored inside  combined_inference/results/.
A CSV with all numeric metrics is written to  combined_inference/results/metrics.csv.
"""

import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                        # BTXRD/
MODEL_DIR    = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")

CLS_CKPT  = os.path.join(MODEL_DIR, "classification_student.pth")
SEG_CKPT  = os.path.join(MODEL_DIR, "segmentation_student.pth")

CLASS_NAMES = [
    "giant cell tumor", "multiple osteochondromas", "osteochondroma",
    "osteofibroma", "osteosarcoma", "other bt", "other mt",
    "simple bone cyst", "synovial osteochondroma",
]
LABEL2IDX = {n: i for i, n in enumerate(CLASS_NAMES)}


# ═══════════════════════════════════════════════════════════════════════════════
# Model Definitions (self-contained so we don't import from KD repos)
# ═══════════════════════════════════════════════════════════════════════════════

class ClassificationStudent(nn.Module):
    """ConvNeXt-Tiny student for 9-class bone-tumor classification."""
    def __init__(self, num_classes=9):
        super().__init__()
        import timm
        self.model = timm.create_model("convnext_tiny", pretrained=False,
                                       num_classes=num_classes)
        self.feature_dim = self.model.head.fc.in_features

    def forward(self, x):
        return self.model(x)


class SegmentationStudent(nn.Module):
    """SegFormer-B2 student for binary tumour segmentation."""
    def __init__(self, num_classes=1, image_size=224):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        config = SegformerConfig.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512")
        config.num_labels = num_classes
        self.model = SegformerForSemanticSegmentation(config)
        self.image_size = image_size

    def forward(self, x):
        input_size = x.shape[2:]
        out = self.model(x, return_dict=True)
        logits = F.interpolate(out.logits, size=input_size,
                               mode="bilinear", align_corners=False)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """Lightweight Grad-CAM for any conv-based model."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = self.gradients = None
        self._hooks = [
            target_layer.register_forward_hook(self._fwd),
            target_layer.register_full_backward_hook(self._bwd),
        ]

    def _fwd(self, m, i, o):  self.activations = o.detach()
    def _bwd(self, m, gi, go): self.gradients = go[0].detach()

    @torch.enable_grad()
    def __call__(self, tensor, target):
        """Return H×W numpy CAM in [0, 1]."""
        self.model.zero_grad()
        out = self.model(tensor)
        scalar = out if out.dim() <= 1 else (
            out[0, target] if out.shape[1] > 1 else out.mean())
        scalar.backward()
        if self.activations is None or self.gradients is None:
            return np.zeros((tensor.shape[2], tensor.shape[3]), np.float32)
        w = self.gradients.cpu().numpy()[0].mean(axis=(1, 2))
        a = self.activations.cpu().numpy()[0]
        cam = np.maximum((w[:, None, None] * a).sum(0), 0)
        if cam.max() > 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (tensor.shape[3], tensor.shape[2]))
        return cam

    def remove(self):
        for h in self._hooks: h.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_cls_model(device):
    model = ClassificationStudent(num_classes=9)
    ckpt = torch.load(CLS_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    # strip 'backbone.' prefix if present
    sd = {(k.replace("backbone.", "") if k.startswith("backbone.") else k): v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def load_seg_model(device, image_size=224):
    model = SegmentationStudent(num_classes=1, image_size=image_size)
    ckpt = torch.load(SEG_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def cls_transform(image_size=384):
    return T.Compose([
        T.Resize(416), T.CenterCrop(image_size), T.ToTensor(),
        T.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def seg_preprocess(image_path, image_size=224):
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - [.485, .456, .406]) / [.229, .224, .225]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, np.array(img)


def dice_score(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    total = p.sum() + g.sum()
    return (2.0 * inter / total) if total > 0 else (1.0 if inter == 0 else 0.0)


def iou_score(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    union = (p | g).sum()
    return (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def find_target_layer(model, keywords, fallback_type=nn.Conv2d):
    """Walk named_modules, return last layer whose name matches any keyword."""
    layer = None
    for name, mod in model.named_modules():
        if any(k in name for k in keywords) and isinstance(mod, fallback_type):
            layer = mod
    if layer is None:  # ultimate fallback: last Conv2d
        for _, mod in model.named_modules():
            if isinstance(mod, fallback_type):
                layer = mod
    return layer


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def make_overlay(image_np, mask, color=(0, 120, 255), alpha=0.45):
    """Overlay a binary mask on an image."""
    vis = image_np.copy()
    vis[mask > 0] = (np.array(color) * alpha +
                     vis[mask > 0] * (1 - alpha)).astype(np.uint8)
    return vis


def build_figure(image_np, gt_mask, pred_mask, seg_cam, cls_cam,
                 cls_probs, gt_label_idx, cls_names, dice, iou,
                 img_name):
    """
    2-row, 4-col figure.
    Row 1: Original | GT Mask | Predicted Mask | Overlay
    Row 2: Seg Grad-CAM | Cls Grad-CAM | Top-5 Bar | Summary
    """
    pred_idx = int(cls_probs.argmax())
    pred_conf = float(cls_probs[pred_idx])
    correct = pred_idx == gt_label_idx

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # ── Row 1 ─────────────────────────────────────────────────────────────
    # Original
    axes[0, 0].imshow(image_np); axes[0, 0].set_title("Original X-ray", fontsize=13, fontweight="bold"); axes[0, 0].axis("off")

    # Ground-truth mask overlay
    gt_vis = make_overlay(image_np, gt_mask, color=(0, 255, 0))
    axes[0, 1].imshow(gt_vis); axes[0, 1].set_title("Ground Truth Mask", fontsize=13, fontweight="bold"); axes[0, 1].axis("off")

    # Predicted mask overlay
    pred_vis = make_overlay(image_np, pred_mask, color=(0, 120, 255))
    axes[0, 2].imshow(pred_vis); axes[0, 2].set_title(f"Predicted Mask\nDice: {dice:.3f}  |  IoU: {iou:.3f}", fontsize=13, fontweight="bold"); axes[0, 2].axis("off")

    # Comparison overlay  (TP=green, FP=blue, FN=red)
    comp = image_np.copy().astype(np.float32)
    tp = (gt_mask > 0) & (pred_mask > 0)
    fp = (gt_mask == 0) & (pred_mask > 0)
    fn = (gt_mask > 0) & (pred_mask == 0)
    comp[tp] = comp[tp] * 0.5 + np.array([0, 220, 0]) * 0.5
    comp[fp] = comp[fp] * 0.5 + np.array([0, 0, 255]) * 0.5
    comp[fn] = comp[fn] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[0, 3].imshow(comp.astype(np.uint8)); axes[0, 3].set_title("Overlay  (G:TP  B:FP  R:FN)", fontsize=13, fontweight="bold"); axes[0, 3].axis("off")

    # ── Row 2 ─────────────────────────────────────────────────────────────
    # Seg Grad-CAM
    seg_hm = cv2.applyColorMap(np.uint8(255 * seg_cam), cv2.COLORMAP_JET)
    seg_hm = cv2.cvtColor(seg_hm, cv2.COLOR_BGR2RGB)
    seg_overlay = cv2.addWeighted(image_np, 0.55, seg_hm, 0.45, 0)
    axes[1, 0].imshow(seg_overlay); axes[1, 0].set_title("Seg Grad-CAM", fontsize=13, fontweight="bold"); axes[1, 0].axis("off")

    # Cls Grad-CAM
    cls_hm = cv2.applyColorMap(np.uint8(255 * cls_cam), cv2.COLORMAP_JET)
    cls_hm = cv2.cvtColor(cls_hm, cv2.COLOR_BGR2RGB)
    cls_overlay = cv2.addWeighted(image_np, 0.55, cls_hm, 0.45, 0)
    axes[1, 1].imshow(cls_overlay); axes[1, 1].set_title("Cls Grad-CAM", fontsize=13, fontweight="bold"); axes[1, 1].axis("off")

    # Top-5 bar chart
    top5 = torch.topk(torch.tensor(cls_probs), min(5, len(cls_names)))
    colors = ["green" if i == gt_label_idx else "steelblue" for i in top5.indices]
    axes[1, 2].barh(range(len(top5.values)), top5.values.numpy(), color=colors)
    axes[1, 2].set_yticks(range(len(top5.values)))
    axes[1, 2].set_yticklabels([cls_names[i] for i in top5.indices], fontsize=11)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_xlabel("Probability", fontsize=11)
    axes[1, 2].set_title("Top-5 Predictions", fontsize=13, fontweight="bold")
    for i, (p, _) in enumerate(zip(top5.values, top5.indices)):
        axes[1, 2].text(p + 0.02, i, f"{p:.1%}", va="center", fontsize=10)

    # Summary card
    ax = axes[1, 3]; ax.axis("off")
    bg = "honeydew" if correct else "mistyrose"
    ax.add_patch(FancyBboxPatch((.03, .1), .94, .8, boxstyle="round,pad=0.03",
                                facecolor=bg, edgecolor="gray", linewidth=1.5,
                                transform=ax.transAxes))
    summary = (
        f"Image:  {img_name}\n\n"
        f"Predicted:  {cls_names[pred_idx]}\n"
        f"Confidence: {pred_conf:.1%}\n\n"
        f"Ground Truth: {cls_names[gt_label_idx]}\n"
        f"Status: {'CORRECT' if correct else 'INCORRECT'}\n\n"
        f"Seg Dice: {dice:.3f}\n"
        f"Seg IoU:  {iou:.3f}"
    )
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=12,
            va="center", ha="center", family="monospace")
    ax.set_title("Summary", fontsize=13, fontweight="bold")

    status_icon = "[CORRECT]" if correct else "[INCORRECT]"
    fig.suptitle(
        f"{cls_names[pred_idx]}  ({pred_conf:.0%})  {status_icon}   |   "
        f"Dice {dice:.3f}   IoU {iou:.3f}",
        fontsize=15, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load models ───────────────────────────────────────────────────────
    print("Loading classification student (ConvNeXt-Tiny)...")
    cls_model = load_cls_model(device)
    print("Loading segmentation student (SegFormer-B2)...")
    seg_model = load_seg_model(device, image_size=args.seg_size)
    print("Models loaded.\n")

    # ── Grad-CAM targets ─────────────────────────────────────────────────
    cls_target = find_target_layer(cls_model, ["stages.3"], nn.Conv2d)
    seg_target = find_target_layer(seg_model, ["decode_head"], nn.Conv2d)

    # ── build sample list ─────────────────────────────────────────────────
    cls_df = pd.read_csv(args.cls_csv)
    cls_df["filename"] = cls_df["image_path"].apply(os.path.basename)
    mask_dir = args.mask_dir

    samples = []
    for _, row in cls_df.iterrows():
        fn = row["filename"]
        mask_fn = fn.replace(".jpeg", "_mask.png").replace(".jpg", "_mask.png")
        mask_path = os.path.join(mask_dir, mask_fn)
        if os.path.exists(row["image_path"]) and os.path.exists(mask_path):
            samples.append({
                "image_path": row["image_path"],
                "mask_path": mask_path,
                "label": row["labels"],
                "filename": fn,
            })
    print(f"Found {len(samples)} samples with both label + mask.\n")

    # ── transforms ────────────────────────────────────────────────────────
    cls_tf = cls_transform(args.cls_size)

    # ── run ────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    vis_dir = os.path.join(RESULTS_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    records = []
    cls_correct = 0

    for idx, s in enumerate(tqdm(samples, desc="Inference")):
        img_pil = Image.open(s["image_path"]).convert("RGB")
        gt_label_idx = LABEL2IDX[s["label"]]

        # ── Classification ────────────────────────────────────────────────
        cls_tensor = cls_tf(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = cls_model(cls_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_conf = float(probs[pred_idx])
        is_correct = pred_idx == gt_label_idx
        cls_correct += int(is_correct)

        # Cls Grad-CAM
        gc_cls = GradCAM(cls_model, cls_target)
        cls_cam = gc_cls(cls_tensor, pred_idx)
        gc_cls.remove()

        # ── Segmentation ─────────────────────────────────────────────────
        seg_tensor, seg_img_np = seg_preprocess(s["image_path"], args.seg_size)
        seg_tensor = seg_tensor.to(device)
        with torch.no_grad():
            seg_logits = seg_model(seg_tensor)
            seg_prob = torch.sigmoid(seg_logits)
            pred_mask = (seg_prob > args.threshold).cpu().numpy()[0, 0].astype(np.uint8)

        gt_mask_pil = Image.open(s["mask_path"]).convert("L").resize(
            (args.seg_size, args.seg_size), Image.NEAREST)
        gt_mask = (np.array(gt_mask_pil) > 127).astype(np.uint8)

        d = dice_score(pred_mask, gt_mask)
        iou = iou_score(pred_mask, gt_mask)

        # Seg Grad-CAM
        gc_seg = GradCAM(seg_model, seg_target)
        seg_cam = gc_seg(seg_tensor, None)  # target ignored for seg (uses .mean())
        gc_seg.remove()

        # ── resize everything to a common display size ────────────────────
        disp = args.seg_size
        img_disp = np.array(img_pil.resize((disp, disp)))
        cls_cam_r = cv2.resize(cls_cam, (disp, disp))
        seg_cam_r = seg_cam  # already at seg_size
        gt_mask_r = gt_mask
        pred_mask_r = pred_mask

        # ── visualisation ─────────────────────────────────────────────────
        if idx < args.max_vis:
            fig = build_figure(
                img_disp, gt_mask_r, pred_mask_r,
                seg_cam_r, cls_cam_r, probs,
                gt_label_idx, CLASS_NAMES, d, iou,
                s["filename"],
            )
            out_name = s["filename"].replace(".jpeg", "").replace(".jpg", "").replace(".png", "")
            fig.savefig(os.path.join(vis_dir, f"{out_name}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        records.append({
            "filename": s["filename"],
            "ground_truth": s["label"],
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": pred_conf,
            "cls_correct": is_correct,
            "dice": d,
            "iou": iou,
        })

    # ── metrics CSV ───────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)

    # ── summary ───────────────────────────────────────────────────────────
    n = len(df)
    acc = cls_correct / n if n else 0
    avg_dice = df["dice"].mean()
    avg_iou  = df["iou"].mean()

    print(f"\n{'='*70}")
    print(f"  COMBINED INFERENCE COMPLETE  ({n} samples)")
    print(f"{'='*70}")
    print(f"  Classification accuracy : {acc:.2%}  ({cls_correct}/{n})")
    print(f"  Segmentation avg Dice   : {avg_dice:.4f}")
    print(f"  Segmentation avg IoU    : {avg_iou:.4f}")
    print()

    # Per-class
    print("  Per-class Classification:")
    for cls in CLASS_NAMES:
        sub = df[df["ground_truth"] == cls]
        if len(sub):
            c = sub["cls_correct"].sum()
            print(f"    {cls:30s}  {c}/{len(sub)}  ({c/len(sub):.0%})")

    print()
    print("  Per-class Segmentation:")
    for cls in CLASS_NAMES:
        sub = df[df["ground_truth"] == cls]
        if len(sub):
            print(f"    {cls:30s}  Dice={sub['dice'].mean():.3f}  IoU={sub['iou'].mean():.3f}  (n={len(sub)})")

    print(f"\n  Metrics CSV  : {csv_path}")
    print(f"  Visualisations: {vis_dir}  ({min(n, args.max_vis)} PNGs)")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Combined Cls + Seg inference")
    p.add_argument("--cls-csv", default=os.path.join(
        PROJECT_ROOT, "augmented_classification_data", "augmented_test.csv"))
    p.add_argument("--mask-dir", default=os.path.join(
        PROJECT_ROOT, "segmentation_masks"))
    p.add_argument("--cls-size", type=int, default=384)
    p.add_argument("--seg-size", type=int, default=224)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-vis", type=int, default=187,
                   help="Max visualisations to save (default: all)")
    main_args = p.parse_args()
    run(main_args)


if __name__ == "__main__":
    main()
