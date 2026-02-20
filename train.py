import os
import pickle
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np

print("Initializing PaddleOCR (CPU mode)...")
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)
print("PaddleOCR initialized successfully!\n")


def create_side_by_side_result(img_array, boxes, texts, scores):
    from PIL import ImageDraw, ImageFont

    colors = [
        (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
    ]

    # Draw bounding boxes
    img = img_array.copy()
    for idx, box in enumerate(boxes):
        box = np.array(box).astype(np.int32)
        cv2.polylines(img, [box], True, colors[idx % len(colors)], 3)

    img_pil = Image.fromarray(img)
    img_w, img_h = img_pil.size

    # Create text panel
    text_w = max(600, img_w)
    text_panel = Image.new('RGB', (text_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(text_panel)

    try:
        font = ImageFont.truetype("fonts/simfang.ttf", 16)
        font_title = ImageFont.truetype("fonts/simfang.ttf", 20)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_title = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
            font_title = ImageFont.load_default()

    y = 20
    draw.text((10, y), "Detected Text:", fill=(0, 0, 0), font=font_title)
    y += 40

    for i, (text, score) in enumerate(zip(texts, scores), 1):
        draw.text((10, y), f"{i}: {text}    {score:.3f}", fill=(0, 0, 0), font=font)
        y += 25
        if y > img_h - 30:
            break

    combined = Image.new('RGB', (img_w + text_w, img_h), (255, 255, 255))
    combined.paste(img_pil, (0, 0))
    combined.paste(text_panel, (img_w, 0))

    return combined


def process_image(image_path, output_dir, drug_name):
    rel_key = f"\\{drug_name}\\{image_path.name}"
    print(f"Processing: {image_path.name}")

    try:
        result = ocr.ocr(str(image_path), cls=True)
        if result is None or result[0] is None:
            print(f"  No text detected in {image_path.name}\n")
            return rel_key, [], None

        boxes  = [line[0] for line in result[0]]
        texts  = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        print(f"  Detected {len(texts)} text regions:")
        for i, (text, score) in enumerate(zip(texts, scores), 1):
            print(f"    {i}: {text}    {score:.3f}")

        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        im_show = create_side_by_side_result(img, boxes, texts, scores)

        output_path = output_dir / f"result_{image_path.name}"
        im_show.save(output_path)
        print(f"  Result saved to: {output_path}\n")

        return rel_key, texts, str(output_path.resolve())

    except Exception as e:
        print(f"  ERROR processing {image_path.name}: {str(e)}\n")
        return rel_key, [], None


def main():
    base_input_dir  = Path("sample obat")
    base_output_dir = Path("output")

    if not base_input_dir.exists():
        print(f"ERROR: Input directory '{base_input_dir}' not found!")
        return

    drug_dirs = [d for d in base_input_dir.iterdir() if d.is_dir()]
    if not drug_dirs:
        print(f"No drug subdirectories found in '{base_input_dir}'!")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif',
                        '.JPG', '.JPEG', '.PNG'}

    total_images  = 0
    total_folders = len(drug_dirs)

    all_sentences     = {}   # {abs_image_path: [ocr_texts]}
    all_result_images = {}   # {abs_image_path: result_image_path}

    print(f"Found {total_folders} drug folder(s) to process\n")
    print("=" * 60)

    for drug_dir in sorted(drug_dirs):
        drug_name  = drug_dir.name
        output_dir = base_output_dir / drug_name
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in drug_dir.iterdir()
                       if f.is_file() and f.suffix in image_extensions]

        if not image_files:
            print(f"[{drug_name}] No images found, skipping...\n")
            continue

        print(f"[{drug_name}] Processing {len(image_files)} image(s)...")
        print("-" * 60)

        for image_path in image_files:
            result = process_image(image_path, output_dir, drug_name)
            if result:
                img_path_str, ocr_texts, result_img_path = result
                all_sentences[img_path_str] = ocr_texts
                if result_img_path:
                    all_result_images[img_path_str] = result_img_path
            total_images += 1

        print(f"[{drug_name}] Done! Results saved to '{output_dir}'\n")

    # Simpan pickle
    pickle_path = base_output_dir / "ocr_results.pkl"
    pickle_data = {
        "images":    all_result_images,
        "sentences": all_sentences,
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(pickle_data, f)

    print("=" * 60)
    print(f"All done! Processed {total_images} image(s) across {total_folders} folder(s).")
    print(f"Results saved under '{base_output_dir}/' folder.")
    print(f"Pickle saved to: {pickle_path}")
    print(f"  - images   : {len(all_result_images)} entries")
    print(f"  - sentences: {len(all_sentences)} entries")


if __name__ == "__main__":
    main()
