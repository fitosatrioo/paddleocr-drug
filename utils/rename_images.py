import os
import csv
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent / "sample obat"
DATASET_CSV = Path(__file__).parent.parent / "dataset.csv"

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}


def main():
    if not BASE_DIR.exists():
        print(f"ERROR: '{BASE_DIR}' not found!")
        return

    drug_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    if not drug_dirs:
        print("No drug subdirectories found!")
        return

    csv_rows = []
    counter = 1

    print("=" * 60)
    print("RENAMING IMAGES")
    print("=" * 60)

    for drug_dir in drug_dirs:
        drug_name = drug_dir.name
        image_files = sorted([
            f for f in drug_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not image_files:
            print(f"[{drug_name}] No images, skipping...")
            continue

        # Pass 1: rename ke temp names (hindari conflict)
        temp_files = []
        for img in image_files:
            temp_name = f"___temp_{counter}{img.suffix}"
            temp_path = drug_dir / temp_name
            img.rename(temp_path)
            temp_files.append((temp_path, counter, img.suffix))
            counter += 1

        # Pass 2: rename dari temp ke final names
        start_num = temp_files[0][1]
        for temp_path, num, ext in temp_files:
            new_name = f"image_{num}{ext}"
            new_path = drug_dir / new_name
            temp_path.rename(new_path)

            relative_path = f"\\{drug_name}\\{new_name}"
            csv_rows.append({'Image Name': relative_path, 'Label': drug_name})

        end_num = temp_files[-1][1]
        print(f"[{drug_name}] image_{start_num} → image_{end_num} ({len(temp_files)} files)")

    print(f"\nTotal: {counter - 1} images renamed")

    # Tulis dataset.csv baru (tanpa kutip dua)
    print(f"\nWriting {DATASET_CSV} ...")
    with open(DATASET_CSV, 'w', encoding='utf-8') as f:
        f.write("Image Name,Label\n")
        for row in csv_rows:
            f.write(f"{row['Image Name']},{row['Label']}\n")

    print(f"  Written {len(csv_rows)} rows to {DATASET_CSV}")
    print("\nDone!")


if __name__ == "__main__":
    main()
