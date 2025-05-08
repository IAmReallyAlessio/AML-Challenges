from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import os
import torch
from tqdm import tqdm
import pandas as pd


# === CONFIGURATION ===
TEST_FOLDER = "/kaggle/input/test-set-v2/test"  # folder with test images
OUTPUT_CSV = "vqa_preds.csv"
QUESTION = "Does this aerial image contain a cactus? Answer with yes or no."


# === LOAD MODEL ===
print("Loading model...")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === FUNCTION TO PREDICT USING VQA ===

def predict_cactus_presence(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

    encoding = processor(image, QUESTION, return_tensors="pt").to(device)
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx].lower()

    # print(answer)

    if "yes" in answer:
        return 1
    elif "no" in answer:
        return 0
    else:
        return None  # uncertain or irrelevant


# === LOOP OVER TEST IMAGES ===
print("Generating labels...")
results = []

for filename in tqdm(os.listdir(TEST_FOLDER)):
    path = os.path.join(TEST_FOLDER, filename)
    label = predict_cactus_presence(path)
    if label is not None:
        results.append({"id": filename, "has_cactus": label})
    else:
        print(f"Skipped {filename} (uncertain answer)")

# === SAVE TO CSV ===
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done! {len(df)} labels saved to {OUTPUT_CSV}")
