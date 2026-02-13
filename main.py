from paddleocr import PaddleOCR

# Initialize PaddleOCR with CPU configuration
print("Initializing PaddleOCR...")
ocr = PaddleOCR(
    use_angle_cls=True,  # Detect text orientation
    lang='en'            # English language
)

# Test image from sample obat folder
img_path = "sample obat/image1.jpeg"

print(f"Processing image: {img_path}\n")

# Run OCR using .ocr() method (for PaddleOCR 2.7.x)
result = ocr.ocr(img_path, cls=True)

# Print detected text
if result and result[0]:
    print("Detected text:")
    for idx, line in enumerate(result[0], 1):
        text = line[1][0]
        confidence = line[1][1]
        print(f"{idx}. '{text}' (confidence: {confidence:.2%})")
else:
    print("No text detected")

print("\nDone!")