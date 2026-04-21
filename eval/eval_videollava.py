import os
import json
import csv
import argparse
import numpy as np
import torch
from PIL import Image
import string
from tqdm import tqdm
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration


def load_frames(image_paths, num_frames=8):
    image_paths = [p for p in image_paths if os.path.exists(p)]
    if len(image_paths) == 0:
        raise ValueError("No images found")

    image_paths = sorted(image_paths)

    if len(image_paths) > num_frames:
        idx = np.linspace(0, len(image_paths) - 1, num_frames).astype(int)
        image_paths = [image_paths[i] for i in idx]
    elif len(image_paths) < num_frames:
        image_paths = image_paths + [image_paths[-1]] * (num_frames - len(image_paths))

    frames = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        frames.append(np.array(img))

    return np.stack(frames)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--data_root", required=True)   # path to skyscraper_gdelt_sentinel
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = VideoLlavaProcessor.from_pretrained(args.model_path)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,   # use this only if accelerate is installed
    ).to(device)

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

            prompt = f"USER: {prompt}\nASSISTANT:"
            # print('\nPROMPT:', prompt, '\n')
            #prompt = f"USER: This is a sequence of images capturing the same location at different times: <video> \nDescribe what is occurring in these images using 2-3 sentences.\nASSISTANT:"

            image_paths = []
            for rel_path in example["video"]:
                rel_path = rel_path.replace("skyscraper_gdelt_sentinel/", "")
                image_paths.append(os.path.join(args.data_root, rel_path))

            try:
                frames = load_frames(image_paths)

                inputs = processor(text=prompt, videos=frames, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                torch.cuda.empty_cache()
                
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        use_cache=False,
                    )

                full_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                prediction = full_response.split("ASSISTANT:")[-1].strip()
                prediction = prediction.translate(str.maketrans('', '', string.punctuation))

            except Exception as e:
                prediction = f"ERROR: {e}"

            # print('GT:', ground_truth)
            # print('PRED:', prediction, '\n')
            writer.writerow([example_id, prompt, ground_truth.lower(), prediction.lower()])
            f.flush()
            print(example_id, '|', ground_truth.lower(), '|', prediction.lower())


if __name__ == "__main__":
    main()