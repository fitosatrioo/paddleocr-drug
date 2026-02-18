import os
from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np

print("Initializing PaddleOCR (CPU mode)...")
ocr = PaddleOCR(
    use_angle_cls=True,  # Enable text orientation detection
    lang='en'            # Language: English
)
print("PaddleOCR initialized successfully!\n")

def create_side_by_side_result(img_array, boxes, texts, scores):
    from PIL import ImageDraw, ImageFont
    import random
    
    colors = [
        (255, 0, 0),      # Red
        (0, 0, 255),      # Blue
        (0, 255, 0),      # Green
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring Green
        (255, 0, 128),    # Deep Pink
    ]
    
    # Draw bounding boxes on image with different colors
    img = img_array.copy()
    for idx, box in enumerate(boxes):
        box = np.array(box).astype(np.int32)
        # Use different color for each box
        color = colors[idx % len(colors)]
        cv2.polylines(img, [box], True, color, 3)
    
    # Convert to PIL for easier text rendering
    img_pil = Image.fromarray(img)
    img_width, img_height = img_pil.size
    
    # Create text panel
    text_width = max(600, img_width)  
    text_height = img_height
    text_panel = Image.new('RGB', (text_width, text_height), (255, 255, 255))
    draw = ImageDraw.Draw(text_panel)
    
    try:
        font = ImageFont.truetype("fonts/simfang.ttf", 16)
        font_title = ImageFont.truetype("fonts/simfang.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_title = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
            font_title = ImageFont.load_default()
    
    # Draw text list
    y_offset = 20
    draw.text((10, y_offset), "Detected Text:", fill=(0, 0, 0), font=font_title)
    y_offset += 40
    
    for i, (text, score) in enumerate(zip(texts, scores), 1):
        line = f"{i}: {text}    {score:.3f}"
        draw.text((10, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += 25
        
        if y_offset > text_height - 30:
            break
    
    combined_width = img_width + text_width
    combined = Image.new('RGB', (combined_width, img_height), (255, 255, 255))
    combined.paste(img_pil, (0, 0))
    combined.paste(text_panel, (img_width, 0))
    
    return combined

def process_image(image_path, output_dir):
    print(f"Processing: {image_path.name}")
    
    try:
        result = ocr.ocr(str(image_path), cls=True)
        if result is None or result[0] is None:
            print(f"  No text detected in {image_path.name}\n")
            return
        
        boxes = [line[0] for line in result[0]]
        texts = [line[1][0] for line in result[0]]
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
        
    except Exception as e:
        print(f"  ERROR processing {image_path.name}: {str(e)}\n")

def main():
    input_dir = Path("sample obat/Abemaciclib")
    output_dir = Path("output/Abemaciclib")
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"ERROR: Input directory '{input_dir}' not found!")
        print(f"Please create the directory and add medicine images.")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    if not image_files:
        print(f"No images found in '{input_dir}' directory!")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    print("=" * 50)
    
    for image_path in image_files:
        process_image(image_path, output_dir)
    
    print("=" * 50)
    print(f"Processing complete! Results saved to '{output_dir}' folder")

if __name__ == "__main__":
    main()
