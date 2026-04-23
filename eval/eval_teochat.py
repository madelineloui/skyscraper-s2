import os
import json
import csv
import argparse
import string
import re
import types
import traceback
import numpy as np

import torch
from PIL import Image
from tqdm import tqdm
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from videollava.model.builder import load_pretrained_model
from videollava.mm_utils import get_model_name_from_path
from videollava.eval.inference import run_inference_single
from videollava.constants import DEFAULT_VIDEO_TOKEN

def load_image_paths(image_paths, num_frames=8):
    image_paths = [p for p in image_paths if os.path.exists(p)]
    if len(image_paths) == 0:
        raise ValueError("No images found")
    image_paths = sorted(image_paths)
    if len(image_paths) > num_frames:
        idx = np.linspace(0, len(image_paths) - 1, num_frames).astype(int).tolist()
        image_paths = [image_paths[i] for i in idx]
    return image_paths
    
def encode_images_fixed(self, images):
        image_features = self.get_model().get_image_tower()(images)
        image_features = image_features.to(device="cuda", dtype=torch.float32)  # match projector
        image_features = self.get_model().mm_projector(image_features)
        image_features = image_features.to(device="cuda", dtype=torch.float16)  # back to fp16 for LLM
        return image_features

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

# def load_image_paths(image_paths, max_images=None):
#     image_paths = [p for p in image_paths if os.path.exists(p)]
#     if len(image_paths) == 0:
#         raise ValueError("No images found")

#     image_paths = sorted(image_paths)

#     if max_images is not None and len(image_paths) > max_images:
#         idx = torch.linspace(0, len(image_paths) - 1, max_images).long().tolist()
#         image_paths = [image_paths[i] for i in idx]

#     return image_paths


def mostly_numbers_long(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    digits = sum(c.isdigit() for c in text)
    return len(text) > 50 and digits / max(len(text), 1) > 0.5

def is_valid_date_format(text):
    return re.fullmatch(r"\d{4}-\d{2}-\d{2}, \d{4}-\d{2}-\d{2}", text.strip()) is not None

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
    parser.add_argument("--teochat_path", required=True)
    parser.add_argument("--output_csv", default="results_teochat.csv")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device}!')

    model_name = get_model_name_from_path(args.teochat_path)
    
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path=args.teochat_path,
        model_base=None,
        model_name=model_name,
        load_8bit=args.load_8bit,
        device=device,
    )
    
    # Load towers — they initialize from their own pretrained weights, not the checkpoint
    if hasattr(model.get_model().get_image_tower(), "load_model"):
        model.get_model().get_image_tower().load_model()
    
    if hasattr(model.get_model().get_video_tower(), "load_model"):
        model.get_model().get_video_tower().load_model()
    
    # Move ONLY the towers to cuda/fp16 — don't touch the main model
    # model.get_model().get_image_tower().to(device="cuda", dtype=torch.float16)
    # model.get_model().get_video_tower().to(device="cuda", dtype=torch.float16)

    # Patch encode_images to fix dtype mismatch at projector boundary 
    model.encode_images = types.MethodType(encode_images_fixed, model)
    
    # Pull processors from towers
    image_processor = model.get_model().get_image_tower().image_processor
    video_processor = model.get_model().get_video_tower().video_processor
    
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
                torch.cuda.empty_cache()
                try:
                    # Pass string paths, not PIL images — run_inference_single handles loading
                    used_paths = load_image_paths(image_paths, num_frames=args.num_frames)
                    print('# images:', len(used_paths))

                    torch.cuda.empty_cache()

                    out = run_inference_single(
                        model=model,
                        processor=image_processor,
                        tokenizer=tokenizer,
                        inp=prompt,
                        image_paths=used_paths,  # sorted string paths, no timestamps needed
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )

                    print(f"OUT (attempt {attempt+1}):", out)

                    if isinstance(out, tuple):
                        pred = out[-1]
                    else:
                        pred = out

                    pred = str(pred).strip()

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
                    print(f"attempt {attempt+1} failed: {e}")
                    traceback.print_exc()  # prints full traceback

                    if "probability tensor" in err_msg or "nan" in err_msg or "inf" in err_msg:
                        torch.cuda.empty_cache()
                        continue

                    prediction = f"ERROR: {e}"
                    break

            if prediction is None:
                prediction = "ERROR: max retries exceeded"

            if not prediction.startswith("ERROR:"):
                prediction = prediction.translate(str.maketrans("", "", string.punctuation)).lower()
            else:
                prediction = prediction.lower()
            ground_truth = ground_truth.translate(str.maketrans("", "", string.punctuation)).lower()

            writer.writerow([example_id, prompt, ground_truth, prediction])
            f.flush()

            print(example_id, "|", ground_truth, "|", prediction)


if __name__ == "__main__":
    main()