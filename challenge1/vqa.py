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
# print("Generating labels...")
# results = []

# for filename in tqdm(os.listdir(TEST_FOLDER)):
#     path = os.path.join(TEST_FOLDER, filename)
#     label = predict_cactus_presence(path)
#     if label is not None:
#         results.append({"id": filename, "has_cactus": label})
#     else:
#         print(f"Skipped {filename} (uncertain answer)")

# # === SAVE TO CSV ===
# df = pd.DataFrame(results)
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"\n✅ Done! {len(df)} labels saved to {OUTPUT_CSV}")

# === OUR TEST ===

CACTUS_SET = "archive/test_set/cactus"
NO_CACTUS_SET = "archive/test_set/no_cactus"

correct_1 = 0
total_1 = 0
correct_0 = 0
total_0 = 0

for filename in tqdm(os.listdir(CACTUS_SET)):
    path = os.path.join(CACTUS_SET, filename)
    label = predict_cactus_presence(path)
    if label is not None:
        if label == 1:
            correct_1 += 1
        total_1 += 1
    else:
        print(f"Skipped {filename} (uncertain answer)")

for filename in tqdm(os.listdir(NO_CACTUS_SET)):
    path = os.path.join(NO_CACTUS_SET, filename)
    label = predict_cactus_presence(path)
    if label is not None:
        if label == 0:
            correct_0 += 1
        total_0 += 1
    else:
        print(f"Skipped {filename} (uncertain answer)")


print(f'Accuracy of the network on 1: {correct_1 / total_1:.2%}')
print(f'Accuracy of the network on 0: {correct_0 / total_0:.2%}')
print(f'Accuracy of the network: {(correct_1 + correct_0) / (total_1 + total_0):.2%}')
