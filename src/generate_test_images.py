import cv2
import os
import numpy as np

DATASET_PATH = "dataset1"

def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def perspective_transform(image):
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [int(0.05 * w), int(0.05 * h)],
        [int(0.95 * w), 0],
        [0, int(0.95 * h)],
        [w, h]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h))


def add_gaussian_noise(image, std=20):
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def crop_and_resize(image, ratio):
    h, w = image.shape[:2]
    margin_y = int(((1 - ratio) / 2) * h)
    margin_x = int(((1 - ratio) / 2) * w)
    cropped = image[margin_y:h - margin_y, margin_x:w - margin_x]
    return cv2.resize(cropped, (w, h))



def generate_augmentations(ref_path, out_folder):
    img = cv2.imread(ref_path)
    if img is None:
        print(f"  [ERROR] Tidak bisa baca: {ref_path}")
        return

    os.makedirs(out_folder, exist_ok=True)
    count = 0

    # 1. Rotasi
    for angle in [-30, -15, -10, 10, 15, 30]:
        out = rotate_image(img, angle)
        cv2.imwrite(os.path.join(out_folder, f"rotasi_{angle}.jpg"), out)
        count += 1

    # 2. Brightness
    cv2.imwrite(os.path.join(out_folder, "brightness_gelap.jpg"),
                cv2.convertScaleAbs(img, alpha=0.5, beta=0))
    cv2.imwrite(os.path.join(out_folder, "brightness_normal.jpg"),
                cv2.convertScaleAbs(img, alpha=1.0, beta=0))
    cv2.imwrite(os.path.join(out_folder, "brightness_terang.jpg"),
                cv2.convertScaleAbs(img, alpha=1.4, beta=40))
    count += 3

    # 3. Blur — ditambah blur berat
    cv2.imwrite(os.path.join(out_folder, "blur_ringan.jpg"),
                cv2.GaussianBlur(img, (5, 5), 0))
    cv2.imwrite(os.path.join(out_folder, "blur_sedang.jpg"),
                cv2.GaussianBlur(img, (11, 11), 0))
    cv2.imwrite(os.path.join(out_folder, "blur_berat.jpg"),
                cv2.GaussianBlur(img, (21, 21), 0))
    count += 3

    # 4. Noise
    cv2.imwrite(os.path.join(out_folder, "noise_gaussian.jpg"),
                add_gaussian_noise(img, std=20))
    count += 1

    # 5. Crop + Resize — ditambah crop_50 untuk simulasi crop ekstrem
    for ratio, label in [(0.90, "90"), (0.75, "75"), (0.60, "60"), (0.50, "50")]:
        cv2.imwrite(os.path.join(out_folder, f"crop_{label}.jpg"),
                    crop_and_resize(img, ratio))
        count += 1

    # 6. Perspective
    cv2.imwrite(os.path.join(out_folder, "perspective.jpg"),
                perspective_transform(img))
    count += 1

    print(f" {count} variasi disimpan → {out_folder}")


def main():
    if not os.path.isdir(DATASET_PATH):
        print(f"[ERROR] Folder dataset '{DATASET_PATH}' tidak ditemukan!")
        return

    books = sorted(os.listdir(DATASET_PATH))
    processed = 0

    for book in books:
        book_path = os.path.join(DATASET_PATH, book)
        if not os.path.isdir(book_path):
            continue

        ref_path = os.path.join(book_path, "reference", "asli.jpg")
        if not os.path.exists(ref_path):
            print(f"[SKIP] '{book}' — reference/asli.jpg tidak ditemukan")
            continue

        out_folder = os.path.join(book_path, "test")
        print(f"\n[{book}]")
        generate_augmentations(ref_path, out_folder)
        processed += 1

    print(f"\n{'='*50}")
    print(f"Selesai! {processed} buku diproses.")
    print(f"Test images ada di: dataset1/[NamaBuku]/test/")


if __name__ == "__main__":
    main()