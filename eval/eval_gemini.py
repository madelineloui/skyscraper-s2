import os
import json
import csv
import argparse
import re
import string
import traceback
import numpy as np

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


def load_frames(image_paths, max_frames=8):
    image_paths = [p for p in image_paths if os.path.exists(p)]

    if len(image_paths) == 0:
        raise ValueError("No images found")

    image_paths = sorted(image_paths)

    # If there are more than max_frames, evenly subsample.
    # If there are max_frames or fewer, keep them all.
    if max_frames is not None and len(image_paths) > max_frames:
        idx = np.linspace(0, len(image_paths) - 1, max_frames).astype(int).tolist()
        image_paths = [image_paths[i] for i in idx]

    return [Image.open(p).convert("RGB") for p in image_paths]


def get_prompt_and_label(example):
    prompt = ""
    label = ""

    for turn in example["conversations"]:
        if turn["from"] == "human":
            prompt = turn["value"]
        elif turn["from"] == "gpt":
            label = turn["value"]

    if prompt == "":
        raise ValueError("No human prompt found")

    return prompt, label


def mostly_numbers_long(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    digits = sum(c.isdigit() for c in text)
    return len(text) > 50 and digits / max(len(text), 1) > 0.5


def is_valid_date_format(text):
    return re.fullmatch(
        r"\d{4}-\d{2}-\d{2}, \d{4}-\d{2}-\d{2}",
        text.strip()
    ) is not None


def is_yes_no(text):
    text = text.translate(str.maketrans("", "", string.punctuation)).strip().lower()
    return text in {"yes", "no"}


def is_less_than_5_words(text):
    return len(text.strip().split()) < 5


def has_number(text):
    return any(c.isdigit() for c in text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", default="google/gemma-3-4b-it")
    parser.add_argument("--output_csv", default="results_gemma.csv")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}!")

    processor = AutoProcessor.from_pretrained(args.model_path)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    existing_ids = set()
    if os.path.exists(args.output_csv):
        with open(args.output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(int(row["id"]))

    file_exists = os.path.exists(args.output_csv)

    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["id", "question", "ground_truth", "prediction"])

        for example in tqdm(data):
            example_id = example["id"]

            if example_id in existing_ids:
                continue

            prompt, ground_truth = get_prompt_and_label(example)

            image_paths = []
            for rel_path in example["video"]:
                rel_path = rel_path.replace("skyscraper_gdelt_sentinel/", "")
                image_paths.append(os.path.join(args.data_root, rel_path))

            prediction = None

            for attempt in range(args.max_retries):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    frames = load_frames(image_paths, max_frames=args.max_frames)

                    messages = [
                        {
                            "role": "user",
                            "content": (
                                [{"type": "image"} for _ in frames]
                                + [{"type": "text", "text": prompt}]
                            ),
                        }
                    ]

                    text = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )

                    inputs = processor(
                        text=text,
                        images=frames,
                        return_tensors="pt",
                    ).to(model.device)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                        )

                    generated = output_ids[0, inputs["input_ids"].shape[1]:]
                    pred = processor.decode(
                        generated,
                        skip_special_tokens=True
                    ).strip()

                    print(f"OUT (attempt {attempt + 1}):", pred)

                    bad_output = (
                        pred == "" or
                        "error" in pred.lower() or
                        re.search(r"(?:\d\s+){3,}\d", pred) or
                        mostly_numbers_long(pred)
                    )

                    if "grounding" in args.json_path:
                        bad_output = bad_output or not is_valid_date_format(pred)

                    if "detection" in args.json_path:
                        bad_output = bad_output or not is_yes_no(pred)

                    if "description" in args.json_path:
                        bad_output = bad_output or is_less_than_5_words(pred)

                    if "classification" in args.json_path:
                        bad_output = bad_output or has_number(pred)

                    if not bad_output:
                        prediction = pred
                        break

                except Exception as e:
                    err_msg = str(e).lower()
                    print(f"attempt {attempt + 1} failed: {e}")
                    traceback.print_exc()

                    if (
                        "probability tensor" in err_msg or
                        "nan" in err_msg or
                        "inf" in err_msg or
                        "out of memory" in err_msg
                    ):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    prediction = f"ERROR: {e}"
                    break

            if prediction is None:
                prediction = "ERROR: max retries exceeded"

            if not prediction.startswith("ERROR:"):
                prediction = prediction.translate(
                    str.maketrans("", "", string.punctuation)
                ).lower()
            else:
                prediction = prediction.lower()

            ground_truth = ground_truth.translate(
                str.maketrans("", "", string.punctuation)
            ).lower()

            writer.writerow([example_id, prompt, ground_truth, prediction])
            f.flush()

            print(example_id, "|", ground_truth, "|", prediction)


if __name__ == "__main__":
    main()
    
    
# import os
# import json
# import csv
# import argparse
# import string
# import time
# import re
# from PIL import Image
# from tqdm import tqdm
# from google import genai
# from google.genai import types

# # Specify project and location
# GCP_PROJECT = ""
# GCP_LOCATION = ""

# # Initialize the client to run on Vertex AI API
# client = genai.Client(
#     vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION
# )

# # --- VALIDATION HELPERS ---

# def is_valid_date_format(text):
#     return re.fullmatch(r"\d{4}-\d{2}-\d{2}, \d{4}-\d{2}-\d{2}", text.strip()) is not None

# def is_yes_no(text):
#     clean = text.translate(str.maketrans("", "", string.punctuation)).strip().lower()
#     return clean in {"yes", "no"}

# def get_class_options(prompt):
#     match = re.search(r"classes:\s*(.*(?:\.|\?|!|$))", prompt, re.IGNORECASE)
#     if match:
#         options_str = match.group(1).rstrip('.?! ')
#         options = [opt.strip().lower() for opt in options_str.split(',')]
#         return options
#     return []

# def get_prompt_and_label(example):
#     prompt, label = "", ""
#     for turn in example["conversations"]:
#         if turn["from"] == "human": prompt = turn["value"]
#         elif turn["from"] == "gpt": label = turn["value"]
#     return prompt, label

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--json_path", required=True)
#     parser.add_argument("--data_root", required=True)
#     parser.add_argument("--output_csv", default="results_gemini.csv")
#     parser.add_argument("--max_retries", type=int, default=5)
#     parser.add_argument("--max_output_tokens", type=int, default=1000)
#     args = parser.parse_args()

#     with open(args.json_path, "r") as f:
#         data = json.load(f)

#     # REFINED RESUME LOGIC: Only skip if 'error' is NOT in the prediction
#     successful_ids = set()
#     if os.path.exists(args.output_csv):
#         with open(args.output_csv, "r", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 try:
#                     # If 'error' is in the prediction, we do NOT add it to successful_ids
#                     # so that it gets processed again.
#                     if "error" not in row["prediction"].lower():
#                         successful_ids.add(int(row["id"]))
#                 except:
#                     continue

#     file_exists = os.path.exists(args.output_csv)
#     # Use 'a' to append, but logic below handles duplicates by checking successful_ids
#     with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(["id", "question", "ground_truth", "prediction"])

#         pbar = tqdm(data)
#         for example in pbar:
#             example_id = example["id"]
            
#             # Skip only if it was processed successfully without errors
#             if example_id in successful_ids: 
#                 continue

#             prompt_text, ground_truth = get_prompt_and_label(example)
            
#             valid_classes = []
#             is_description_task = "description" in args.json_path
#             if "classification" in args.json_path:
#                 valid_classes = get_class_options(prompt_text)

#             # Load images
#             images = []
#             for rel_path in example["video"]:
#                 full_path = os.path.join(args.data_root, rel_path.replace("skyscraper_gdelt_sentinel/", ""))
#                 if os.path.exists(full_path):
#                     images.append(Image.open(full_path).convert("RGB"))

#             if not images:
#                 tqdm.write(f"ID {example_id}: No images found.")
#                 continue

#             prediction = "error: max retries exceeded"
            
#             for attempt in range(args.max_retries):
#                 try:
#                     temp = 0.1
                    
#                     response = client.models.generate_content(
#                         model="gemini-2.5-flash",
#                         contents=images + [prompt_text],
#                         config=types.GenerateContentConfig(
#                             temperature=temp, 
#                             max_output_tokens=args.max_output_tokens,
#                         )
#                     )

#                     if not response.text:
#                         reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
#                         raise ValueError(f"Empty response. Reason: {reason}")

#                     pred_raw = response.text.strip()

#                     # --- TASK SPECIFIC VALIDATION ---
#                     if "grounding" in args.json_path and not is_valid_date_format(pred_raw):
#                         raise ValueError(f"Grounding Format Error: {pred_raw}")
                    
#                     if is_description_task:
#                         if not pred_raw[-1] in ".?!":
#                             raise ValueError(f"Description seems cut off: {pred_raw[-1]}")
#                         prediction = pred_raw
                    
#                     else:
#                         pred_clean = pred_raw.translate(str.maketrans("", "", string.punctuation)).lower().strip()
                        
#                         if "detection" in args.json_path and not is_yes_no(pred_clean):
#                             raise ValueError(f"Detection Error: {pred_clean}")

#                         if "classification" in args.json_path and valid_classes:
#                             if pred_clean not in valid_classes:
#                                 raise ValueError(f"Invalid class: '{pred_clean}'. Must be one of {valid_classes}")
                        
#                         prediction = pred_clean

#                     break 

#                 except Exception as e:
#                     tqdm.write(f"ID {example_id} | Attempt {attempt+1} failed: {e}")
#                     time.sleep(1)

#             # Write Result
#             gt_log = ground_truth if is_description_task else ground_truth.translate(str.maketrans("", "", string.punctuation)).lower().strip()
            
#             writer.writerow([example_id, prompt_text, ground_truth, prediction])
#             f.flush()
#             tqdm.write(f"ID: {example_id} | GT: {gt_log[:50]}... | PRED: {prediction[:50]}...")

# if __name__ == "__main__":
#     main()