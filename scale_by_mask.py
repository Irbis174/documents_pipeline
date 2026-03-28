import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def imread_robust(path: Path, flags: int) -> np.ndarray | None:
    """Надёжное чтение на Windows (в т.ч. кириллица в пути)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_robust_png(path: Path, img: np.ndarray) -> bool:
    """Надёжная запись PNG (через imencode + tofile)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode('.png', img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочить 4 точки: [tl, tr, br, bl]."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_object_by_mask(image: np.ndarray, mask: np.ndarray, padding: float = 0.0):
    """
    По маске:
    - берём самый большой контур
    - строим minAreaRect (повёрнутый прямоугольник)
    - делаем perspective warp так, чтобы объект занял весь кадр результата
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_bin = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('Mask is empty: no non-zero pixels.')

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)   # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = _order_points(box)

    if padding > 0:
        c = box.mean(axis=0, keepdims=True)
        box = c + (box - c) * (1.0 + 2.0 * padding)

    (tl, tr, br, bl) = box
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    width = max(width, 1)
    height = max(height, 1)

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_CUBIC)

    info = {
        'method': 'min_area_rect',
        'min_area_rect': {
            'center': [float(rect[0][0]), float(rect[0][1])],
            'size': [float(rect[1][0]), float(rect[1][1])],
            'angle_deg': float(rect[2]),
            'corners_src': box.tolist(),
        },
        'transform': {'perspective_matrix_3x3': M.tolist()},
        'output_native': {'width': int(width), 'height': int(height)},
    }
    return warped, info


def resize_to_screen(img: np.ndarray, screen_wh: tuple[int, int], mode: str = 'fit'):
    """
    mode:
      - fit: вписать без обрезки (поля по краям)
      - fill: заполнить весь экран (возможна обрезка)
      - stretch: растянуть (может исказить пропорции)
    """
    sw, sh = screen_wh
    h, w = img.shape[:2]

    if mode == 'stretch':
        out = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_CUBIC)
        return out, {'mode': 'stretch', 'scale_x': sw / w, 'scale_y': sh / h, 'offset_x': 0, 'offset_y': 0}

    scale_fit = min(sw / w, sh / h)
    scale_fill = max(sw / w, sh / h)
    scale = scale_fit if mode == 'fit' else scale_fill

    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

    if mode == 'fit':
        canvas = np.zeros((sh, sw, img.shape[2]), dtype=img.dtype)
        ox = (sw - nw) // 2
        oy = (sh - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas, {'mode': 'fit', 'scale': scale, 'offset_x': ox, 'offset_y': oy, 'crop': None}

    ox = (nw - sw) // 2
    oy = (nh - sh) // 2
    cropped = resized[oy:oy + sh, ox:ox + sw]
    return cropped, {'mode': 'fill', 'scale': scale, 'offset_x': -ox, 'offset_y': -oy, 'crop': [ox, oy, ox + sw, oy + sh]}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def collect_images(images_in: Path, recursive: bool) -> list[Path]:
    if images_in.is_file():
        return [images_in.resolve()]
    if images_in.is_dir():
        if recursive:
            return [p.resolve() for p in images_in.rglob('*') if is_image_file(p)]
        return [p.resolve() for p in images_in.iterdir() if is_image_file(p)]
    raise FileNotFoundError(f'Images path not found: {images_in}')


def strip_parts(rel: Path, strip_dirs: list[str]) -> Path:
    if not strip_dirs:
        return rel
    parts = [p for p in rel.parts if p not in strip_dirs]
    return Path(*parts) if parts else Path(rel.name)


def map_mask_path(
    img_path: Path,
    images_root: Path,
    masks_root: Path,
    strip_dirs: list[str],
    mask_suffix: str,
    mask_ext: str,
) -> Path:
    """
    rel = img_path относительно images_root (с учётом strip_dirs)
    mask лежит в masks_root/<rel_parent>/<stem + mask_suffix + mask_ext>
    """
    rel = img_path.relative_to(images_root)
    rel = strip_parts(rel, strip_dirs)
    rel_parent = rel.parent
    stem = img_path.stem
    return (masks_root / rel_parent / f'{stem}{mask_suffix}{mask_ext}').resolve()


def out_path_for_image(
    img_path: Path,
    images_root: Path,
    out_root: Path,
    strip_dirs: list[str],
    flat: bool,
    out_suffix: str,
) -> Path:
    """
    По умолчанию: out_root/<rel_parent>/<stem + out_suffix>.png
    rel_parent повторяет структуру images_root (с учётом strip_dirs).
    """
    rel = img_path.relative_to(images_root)
    rel = strip_parts(rel, strip_dirs)

    if flat:
        safe_name = '__'.join(rel.parts).replace(':', '')
        safe_stem = Path(safe_name).with_suffix('').name
        return (out_root / f'{safe_stem}{out_suffix}.png').resolve()

    rel_parent = rel.parent
    stem = img_path.stem
    return (out_root / rel_parent / f'{stem}{out_suffix}.png').resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='Путь к изображению ИЛИ к папке с изображениями')
    ap.add_argument('--masks', required=True, help='Путь к маске ИЛИ к папке с масками')
    ap.add_argument('--out', default='warped_out', help='Папка для сохранения результатов')

    # ВАЖНО: по умолчанию рекурсивно для папок
    ap.add_argument('--recursive', action='store_true', help='Рекурсивно обходить папку с изображениями')
    ap.add_argument('--no-recursive', action='store_true', help='Не рекурсивно (только верхний уровень)')

    ap.add_argument('--strip-dir', action='append', default=[], help='Удалить сегмент пути (можно несколько раз)')
    ap.add_argument('--mask-suffix', default='_mask', help='Суффикс файла маски (по умолчанию "_mask")')
    ap.add_argument('--mask-ext', default='.png', help='Расширение файла маски (по умолчанию .png)')

    ap.add_argument('--padding', type=float, default=0.02)
    ap.add_argument('--screen-w', type=int, default=1920)
    ap.add_argument('--screen-h', type=int, default=1080)
    ap.add_argument('--mode', choices=['fit', 'fill', 'stretch'], default='fit')

    ap.add_argument('--flat', action='store_true', help='Сохранять в одну плоскую папку')
    ap.add_argument('--save-json', action='store_true', help='Сохранять json рядом с результатом')
    ap.add_argument('--out-suffix', default='_warped', help='Суффикс выходного файла')

    args = ap.parse_args()

    images_in = Path(args.images).resolve()
    masks_in = Path(args.masks).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mask_ext = args.mask_ext if args.mask_ext.startswith('.') else f'.{args.mask_ext}'
    strip_dirs = [d for d in args.strip_dir if d]  # может быть пусто

    # recursive по умолчанию TRUE для директорий
    if images_in.is_dir():
        recursive = True
        if args.no_recursive:
            recursive = False
        if args.recursive:
            recursive = True
    else:
        recursive = False

    # roots для маппинга
    if images_in.is_file():
        images_root = images_in.parent
        img_list = [images_in]
    else:
        images_root = images_in
        img_list = collect_images(images_in, recursive=recursive)

    if masks_in.is_file():
        masks_root = masks_in.parent
        single_mask_file = masks_in
    else:
        masks_root = masks_in
        single_mask_file = None

    stats = {
        'images_found': len(img_list),
        'processed': 0,
        'skipped_missing_mask': 0,
        'skipped_unreadable_image': 0,
        'skipped_unreadable_mask': 0,
        'skipped_empty_mask': 0,
        'write_failed': 0,
    }

    if len(img_list) == 0:
        print('No images found.')
        print('images_in:', images_in)
        print('recursive:', recursive)
        return

    pbar = tqdm(img_list, desc='Warping')
    for img_path in pbar:
        # mask: либо явно один файл (когда images=файл), либо маппим по relpath
        if single_mask_file is not None and images_in.is_file():
            mask_path = single_mask_file
        else:
            mask_path = map_mask_path(
                img_path=img_path,
                images_root=images_root,
                masks_root=masks_root,
                strip_dirs=strip_dirs,
                mask_suffix=args.mask_suffix,
                mask_ext=mask_ext,
            )

        if not mask_path.exists():
            stats['skipped_missing_mask'] += 1
            pbar.set_postfix(processed=stats['processed'], missing_mask=stats['skipped_missing_mask'], write_fail=stats['write_failed'])
            continue

        bgr = imread_robust(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            stats['skipped_unreadable_image'] += 1
            pbar.set_postfix(processed=stats['processed'], bad_img=stats['skipped_unreadable_image'], write_fail=stats['write_failed'])
            continue

        msk = imread_robust(mask_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            stats['skipped_unreadable_mask'] += 1
            pbar.set_postfix(processed=stats['processed'], bad_mask=stats['skipped_unreadable_mask'], write_fail=stats['write_failed'])
            continue

        try:
            warped, info1 = warp_object_by_mask(bgr, msk, padding=args.padding)
        except ValueError:
            stats['skipped_empty_mask'] += 1
            pbar.set_postfix(processed=stats['processed'], empty_mask=stats['skipped_empty_mask'], write_fail=stats['write_failed'])
            continue

        final, info2 = resize_to_screen(warped, (args.screen_w, args.screen_h), mode=args.mode)
        info = {'warp': info1, 'resize': info2}

        out_path = out_path_for_image(
            img_path=img_path,
            images_root=images_root,
            out_root=out_root,
            strip_dirs=strip_dirs,
            flat=args.flat,
            out_suffix=args.out_suffix,
        )

        ok = imwrite_robust_png(out_path, final)
        if not ok:
            stats['write_failed'] += 1
        else:
            stats['processed'] += 1
            if args.save_json:
                json_path = out_path.with_suffix('.json')
                try:
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    json_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
                except Exception:
                    pass

        pbar.set_postfix(processed=stats['processed'], missing_mask=stats['skipped_missing_mask'], empty_mask=stats['skipped_empty_mask'], write_fail=stats['write_failed'])

    print('--- Done ---')
    for k, v in stats.items():
        print(f'{k}: {v}')
    print('out:', out_root)


if __name__ == '__main__':
    main()