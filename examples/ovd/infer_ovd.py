import argparse
import html
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL = "omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "assets" / "person.jpg"
DEFAULT_LABELS = "person"


def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM-R1 open-vocabulary detection on one image.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path.")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Comma-separated object names, e.g. 'drink,fruit'.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ovd_demo"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default=None, help="Optional attention backend, e.g. flash_attention_2.")
    parser.add_argument(
        "--max-memory",
        default=None,
        help="Optional device_map='auto' memory map, e.g. 'cuda:7GiB,cpu:24GiB'.",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Load model files from the local cache only.")
    return parser.parse_args()


def build_prompt(labels):
    label_list = [label.strip() for label in labels.split(",") if label.strip()]
    if not label_list:
        raise ValueError("--labels must include at least one object name.")

    return (
        "First think about the reasoning process in the mind and then provide the user with the answer. "
        "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively. Please carefully check the image and detect the following objects: "
        f"{label_list}. "
        "Output the bbox coordinates of detected objects in <answer></answer>. "
        "The bbox coordinates in Markdown format should be:\n"
        "```json\n"
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "object name"}]\n'
        "```\n"
        'If no targets are detected in the image, simply respond with "None".'
    )


def parse_torch_dtype(name):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def parse_max_memory(value):
    if not value:
        return None

    max_memory = {}
    for item in value.split(","):
        key, memory = item.split(":", 1)
        key = key.strip().lower()
        if key in {"cuda", "gpu", "0"}:
            max_memory[0] = memory.strip()
        elif key == "cpu":
            max_memory["cpu"] = memory.strip()
        else:
            raise ValueError(f"Unsupported max-memory key: {key}")
    return max_memory


def load_model_and_processor(args):
    model_kwargs = {
        "torch_dtype": parse_torch_dtype(args.torch_dtype),
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    max_memory = parse_max_memory(args.max_memory)
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
        if max_memory:
            model_kwargs["max_memory"] = max_memory

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    if args.device in {"cuda", "cpu"}:
        model = model.to(args.device)

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    return model.eval(), processor


def get_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def strip_json_fence(text):
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return fence_match.group(1).strip() if fence_match else text.strip()


def extract_tag(text, tag):
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else ""


def parse_detections(raw_output, image_size):
    answer_text = extract_tag(raw_output, "answer") or raw_output
    answer_text = strip_json_fence(answer_text)
    if not answer_text or answer_text.strip().lower() == "none":
        return []

    data = None
    try:
        data = json.loads(answer_text)
    except json.JSONDecodeError:
        try:
            import json_repair

            data = json_repair.loads(answer_text)
        except Exception:
            data = None

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    width, height = image_size
    detections = []
    for item in data:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox_2d")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(coord) for coord in box]
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "bbox_2d": [round(x1), round(y1), round(x2), round(y2)],
                "label": str(item.get("label", "object")),
            }
        )
    return detections


def draw_detections(image, detections):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    palette = ["#ef4444", "#2563eb", "#16a34a", "#9333ea", "#f97316", "#0891b2"]
    for index, detection in enumerate(detections):
        color = palette[index % len(palette)]
        x1, y1, x2, y2 = detection["bbox_2d"]
        label = detection["label"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)

        label_box = draw.textbbox((x1, y1), label, font=font)
        label_width = label_box[2] - label_box[0] + 10
        label_height = label_box[3] - label_box[1] + 8
        label_y = max(0, y1 - label_height)
        draw.rectangle((x1, label_y, x1 + label_width, label_y + label_height), fill=color)
        draw.text((x1 + 5, label_y + 4), label, fill="white", font=font)
    return result


def run_inference(model, processor, image, prompt, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    ).to(get_input_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]


def write_report(output_dir, image_name, annotated_name, prompt, raw_output, detections):
    rows = "\n".join(
        f"<li><code>{html.escape(item['label'])}</code>: {html.escape(str(item['bbox_2d']))}</li>"
        for item in detections
    )
    rows = rows or "<li>No detections parsed.</li>"
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>VLM-R1 OVD Local Inference</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; color: #172033; background: #f6f7fb; }}
main {{ max-width: 1180px; margin: 0 auto; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
.card {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; }}
img {{ width: 100%; border-radius: 6px; border: 1px solid #e5e7eb; }}
pre {{ white-space: pre-wrap; background: #f1f5f9; padding: 12px; border-radius: 6px; }}
@media (max-width: 840px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<h1>VLM-R1 OVD Local Inference</h1>
<div class="grid">
  <section class="card"><h2>Input</h2><img src="{image_name}" alt="Input image" /></section>
  <section class="card"><h2>Annotated Output</h2><img src="{annotated_name}" alt="Annotated image" /></section>
</div>
<section class="card"><h2>Prompt</h2><pre>{html.escape(prompt)}</pre></section>
<section class="card"><h2>Parsed Detections</h2><ul>{rows}</ul></section>
<section class="card"><h2>Raw Model Output</h2><pre>{html.escape(raw_output)}</pre></section>
</main>
</body>
</html>
"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(args.labels)
    model, processor = load_model_and_processor(args)
    raw_output = run_inference(model, processor, image, prompt, args.max_new_tokens)
    detections = parse_detections(raw_output, image.size)
    annotated = draw_detections(image, detections)

    image_name = args.image.name
    input_copy = args.output_dir / image_name
    image.save(input_copy)
    annotated_name = "annotated.png"
    annotated.save(args.output_dir / annotated_name)
    (args.output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (args.output_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")
    (args.output_dir / "detections.json").write_text(json.dumps(detections, indent=2), encoding="utf-8")
    write_report(args.output_dir, image_name, annotated_name, prompt, raw_output, detections)

    print(f"Parsed {len(detections)} detections")
    print(f"Annotated image: {args.output_dir / annotated_name}")
    print(f"Report: {args.output_dir / 'report.html'}")
    print(raw_output)


if __name__ == "__main__":
    main()
