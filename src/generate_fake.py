import cv2
import os
import numpy as np

DATASET_PATH = "dataset1"


def contrast_tweak(image, alpha=1.8, beta=-60):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_watermark(image, text="COPY"):
    out = image.copy()
    h, w = out.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 150
    thickness  = max(2, int(w / 100))
    text_size  = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    y = (h + text_size[1]) // 2
    cv2.putText(out, text, (x, y), font, font_scale,
                (180, 180, 180), thickness * 5, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), font, font_scale,
                (150, 150, 150), thickness, cv2.LINE_AA)
    return out


def gaussian_blur(image, ksize=15):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def add_noise(image, std=40):
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def crop_center(image, ratio=0.70):
    h, w = image.shape[:2]
    my = max(int(((1 - ratio) / 2) * h), 1)
    mx = max(int(((1 - ratio) / 2) * w), 1)
    cropped = image[my:h - my, mx:w - mx]
    return cv2.resize(cropped, (w, h))


def add_jpeg_artifacts(image, quality=8):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)


# ---- KOMBINASI --------------------------------------------------------------

def generate_fakes(ref_path, out_folder):
    img = cv2.imread(ref_path)
    if img is None:
        print(f"  [ERROR] Tidak bisa baca: {ref_path}")
        return

    os.makedirs(out_folder, exist_ok=True)
    count = 0

    variants = [
        # ---- Grup A: Kontras + Distorsi ------------------------------------
        ("fake_contrast_blur.jpg",
         gaussian_blur(contrast_tweak(img, alpha=2.0, beta=-80), ksize=15)),

        ("fake_contrast_noise.jpg",
         add_noise(contrast_tweak(img, alpha=0.45, beta=40), std=40)),

        # crop diperketat dari 0.70 ke 0.60
        ("fake_contrast_crop.jpg",
         crop_center(contrast_tweak(img, alpha=2.0, beta=-80), ratio=0.60)),

        ("fake_contrast_noise_blur.jpg",
         gaussian_blur(add_noise(contrast_tweak(img, alpha=0.45, beta=40), std=40), ksize=11)),

        # ---- Grup B: Watermark + Degradasi Struktural ----------------------
        ("fake_wm_blur.jpg",
         gaussian_blur(add_watermark(img, text="COPY"), ksize=17)),

        ("fake_wm_crop_noise.jpg",
         add_noise(crop_center(add_watermark(img, text="BAJAKAN"), ratio=0.68), std=40)),

        ("fake_wm_contrast_blur.jpg",
         gaussian_blur(contrast_tweak(add_watermark(img, text="COPY"), alpha=1.9, beta=-70),
                       ksize=15)),

        # ---- Grup C: Crop Sangat Agresif + Noise ---------------------------
        ("fake_crop_noise.jpg",
         add_noise(crop_center(img, ratio=0.65), std=45)),

        ("fake_crop_blur.jpg",
         gaussian_blur(crop_center(img, ratio=0.68), ksize=21)),

        # crop diperketat dari 0.70 ke 0.55, quality turun dari 12 ke 8
        ("fake_crop_jpeg.jpg",
         add_jpeg_artifacts(crop_center(img, ratio=0.55), quality=8)),

        # ---- Grup D: Multi-efek Struktural ---------------------------------
        ("fake_multi_1.jpg",
         crop_center(add_watermark(contrast_tweak(img, alpha=1.9, beta=-70), text="COPY"),
                     ratio=0.72)),

        ("fake_multi_2.jpg",
         add_noise(crop_center(contrast_tweak(img, alpha=0.45, beta=40), ratio=0.68), std=35)),

        ("fake_multi_3.jpg",
         gaussian_blur(add_noise(contrast_tweak(img, alpha=1.8, beta=-60), std=35), ksize=21)),
    ]

    for fname, result in variants:
        cv2.imwrite(os.path.join(out_folder, fname), result)
        count += 1

    print(f"  {count} gambar palsu disimpan -> {out_folder}")


def main():
    if not os.path.isdir(DATASET_PATH):
        print(f"[ERROR] Folder '{DATASET_PATH}' tidak ditemukan!")
        return

    books = sorted(os.listdir(DATASET_PATH))
    processed = 0

    for book in books:
        book_path = os.path.join(DATASET_PATH, book)
        if not os.path.isdir(book_path):
            continue

        ref_path = os.path.join(book_path, "reference", "asli.jpg")
        if not os.path.exists(ref_path):
            print(f"[SKIP] '{book}' -- reference/asli.jpg tidak ditemukan")
            continue

        out_folder = os.path.join(book_path, "fake")
        print(f"\n[{book}]")
        generate_fakes(ref_path, out_folder)
        processed += 1

    print(f"\n{'='*50}")
    print(f"Selesai. {processed} buku diproses.")
    print(f"Gambar palsu ada di: dataset1/[NamaBuku]/fake/")


if __name__ == "__main__":
    main()