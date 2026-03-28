"""predict_class_only.py

Мини-пайплайн: segmentation -> warp(обрезка/выпрямление) -> classification.

Поведение:
- Для каждого входного изображения:
  1) строит маску документа (segmentation checkpoint),
  2) выпрямляет/обрезает документ по маске (warp_object_by_mask),
  3) классифицирует результат,
  4) сохраняет *обрезанное* изображение в папку --out именем класса.
     Если имя занято: добавляет суффикс _1, _2, ...

По умолчанию печатает в stdout только класс (по строке на изображение).

Пример запуска:
python document_pipeline.py `
>> --input 'Passports_clean\7_ua\type4\8_1_39_4737.png' `
>> --seg-ckpt "runs/deeplabv3p/best.pt" `
>> --cls-ckpt "runs/doc_cls_v2/best.pt" `
>> --label2id "runs/doc_cls_v2/label2id.json" `
>> --out "out_pipeline" `
>> --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import timm
import segmentation_models_pytorch as smp
from PIL import Image, ImageOps
from torchvision import transforms

from scale_by_mask import imread_robust, warp_object_by_mask

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def _iter_images(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f'Input not found: {p}')
    return sorted([x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in IMG_EXTS])


def _resize_pad_to_square(rgb: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """LongestMaxSize + PadIfNeeded(center), как в segmentation.py."""
    h, w = rgb.shape[:2]
    scale = float(size) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    out = np.zeros((size, size, 3), dtype=np.uint8)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    meta = {
        'orig_w': w, 'orig_h': h,
        'new_w': new_w, 'new_h': new_h,
        'pad_left': pad_left, 'pad_top': pad_top,
        'size': size,
    }
    return out, meta


def _unpad_resize_mask(mask_sq01: np.ndarray, meta: dict) -> np.ndarray:
    """mask_sq01 (S,S) -> mask_orig (H,W) 0/255."""
    pad_left = int(meta['pad_left'])
    pad_top = int(meta['pad_top'])
    new_w = int(meta['new_w'])
    new_h = int(meta['new_h'])

    cropped = mask_sq01[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    orig_w = int(meta['orig_w'])
    orig_h = int(meta['orig_h'])

    mask = cv2.resize(cropped.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return (mask * 255).astype(np.uint8)


def _sanitize_label(label: str) -> str:
    """Делаем безопасное имя файла из класса."""
    s = label.strip()
    s = s.replace(' ', '_')
    # запретные символы Windows + прочие
    s = re.sub(r'[<>:"/\\|?*\n\r\t]+', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('._ ')
    return s or 'class'


def _next_free_path(out_dir: Path, base: str, ext: str) -> Path:
    """base.ext, если занято -> base_1.ext, base_2.ext, ..."""
    p0 = out_dir / f'{base}{ext}'
    if not p0.exists():
        return p0
    i = 1
    while True:
        pi = out_dir / f'{base}_{i}{ext}'
        if not pi.exists():
            return pi
        i += 1


def _ensure_uint8_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError('Empty image')
    if img.dtype == np.uint8:
        return img
    x = np.asarray(img)
    if x.max() <= 1.0:
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


class Segmenter:
    def __init__(self, ckpt_path: Path, device: torch.device):
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        encoder = ckpt.get('encoder', 'resnet101')
        img_size = int(ckpt.get('img_size', 640))

        model = smp.DeepLabV3Plus(
            encoder_name=str(encoder),
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        model.load_state_dict(ckpt['model'], strict=True)
        model.to(device).eval()

        self.model = model
        self.device = device
        self.img_size = img_size
        # Albumentations Normalize() default (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @torch.no_grad()
    def mask(self, bgr: np.ndarray, thr: float) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sq, meta = _resize_pad_to_square(rgb, self.img_size)

        x = sq.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))  # CHW
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            logits = self.model(xt)[0, 0]  # (S,S)

        probs = torch.sigmoid(logits).float().cpu().numpy()
        mask_sq01 = (probs > float(thr)).astype(np.uint8)
        return _unpad_resize_mask(mask_sq01, meta)


class PadToSquare:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        pad_left = (s - w) // 2
        pad_right = s - w - pad_left
        pad_top = (s - h) // 2
        pad_bottom = s - h - pad_top
        return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))


class Classifier:
    def __init__(self, ckpt_path: Path, label2id_path: Path, device: torch.device):
        ckpt = torch.load(str(ckpt_path), map_location='cpu')

        model = timm.create_model(
            ckpt['model_name'],
            pretrained=False,
            num_classes=int(ckpt['num_classes']),
        )
        model.load_state_dict(ckpt['model_state'], strict=True)
        model.to(device).eval()

        label2id = json.loads(Path(label2id_path).read_text(encoding='utf-8'))
        self.id2label = {int(v): str(k) for k, v in label2id.items()}

        img_size = int(ckpt['img_size'])
        self.tfms = transforms.Compose(
            [
                PadToSquare(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.model = model
        self.device = device

    @torch.no_grad()
    def predict_label(self, bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.tfms(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            logits = self.model(x)[0]  # (C,)

        pred_id = int(torch.argmax(logits).item())
        return self.id2label.get(pred_id, str(pred_id))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True, help='Путь к изображению или папке')
    ap.add_argument('--out', type=str, default='out_cropped', help='Куда сохранять обрезанные изображения')
    ap.add_argument('--ext', type=str, default='.png', help='Расширение сохранения (.png|.jpg|...)')
    ap.add_argument('--seg-ckpt', type=str, required=True, help='best.pt от segmentation.py')
    ap.add_argument('--cls-ckpt', type=str, required=True, help='best.pt от train_classifier.py')
    ap.add_argument('--label2id', type=str, required=True, help='label2id.json от train_classifier.py')
    ap.add_argument('--device', type=str, default='cuda', help='cuda|cpu')
    ap.add_argument('--seg-thr', type=float, default=0.5, help='Порог биноризации маски')
    ap.add_argument('--padding', type=float, default=0.02, help='padding для warp_object_by_mask')
    ap.add_argument('--quiet', action='store_true', help='Ничего не печатать в stdout')
    args = ap.parse_args()

    ext = args.ext.strip()
    if not ext.startswith('.'):
        ext = '.' + ext

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    seg = Segmenter(Path(args.seg_ckpt), device)
    cls = Classifier(Path(args.cls_ckpt), Path(args.label2id), device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = _iter_images(Path(args.input))
    for p in imgs:
        bgr = imread_robust(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        # seg -> warp (если warp не получился, используем исходник, чтобы не падать)
        try:
            mask = seg.mask(bgr, thr=args.seg_thr)
            warped, _ = warp_object_by_mask(bgr, mask, padding=float(args.padding))
            warped = _ensure_uint8_bgr(warped)
        except Exception:
            warped = _ensure_uint8_bgr(bgr)

        label = cls.predict_label(warped)

        # save as "<class>.ext" with counter if exists
        safe = _sanitize_label(label)
        out_path = _next_free_path(out_dir, safe, ext)
        cv2.imwrite(str(out_path), warped)

        if not args.quiet:
            print(label)


if __name__ == '__main__':
    main()
