# VLM-R1 OVD Local Inference

This example runs the released VLM-R1 open-vocabulary detection checkpoint on a local image and writes an annotated result. It mirrors the public Hugging Face Space prompt while making the prompt, JSON parsing, and visualization reusable from the command line.

## Install

Follow the repository setup first. For a minimal inference-only environment, the key packages are:

```bash
pip install torch torchvision transformers accelerate pillow json_repair
```

`json_repair` is optional but recommended because model outputs may contain Markdown fences or slightly malformed JSON.

## Quick Start

```bash
python examples/ovd/infer_ovd.py \
  --image examples/ovd/assets/person.jpg \
  --labels person \
  --output-dir outputs/ovd_person
```

The script writes:

- `annotated.png`: input image with predicted boxes and labels.
- `detections.json`: parsed bounding boxes.
- `raw_output.txt`: raw model response with `<think>` and `<answer>` tags.
- `report.html`: side-by-side input, annotated output, prompt, parsed boxes, and raw output.

## More Examples

```bash
python examples/ovd/infer_ovd.py \
  --image examples/ovd/assets/drinks_fruit.jpg \
  --labels "drink,fruit" \
  --output-dir outputs/ovd_drinks_fruit
```

```bash
python examples/ovd/infer_ovd.py \
  --image examples/ovd/assets/desk.png \
  --labels "keyboard,white cup,laptop" \
  --output-dir outputs/ovd_desk
```

## Gallery

After running multiple examples, build a switchable gallery page:

```bash
python examples/ovd/build_gallery.py \
  --case "Person=outputs/ovd_person" \
  --case "Drinks/Fruit=outputs/ovd_drinks_fruit" \
  --case "Desk=outputs/ovd_desk"
```

Open `outputs/ovd_gallery/index.html` to compare the input image, annotated output, parsed detections, and each case's full report.

## Low-Memory GPUs

The checkpoint is based on Qwen2.5-VL-3B. If your GPU is tight on memory, let `accelerate` offload part of the model:

```bash
python examples/ovd/infer_ovd.py \
  --image examples/ovd/assets/person.jpg \
  --labels person \
  --device auto \
  --max-memory "cuda:7GiB,cpu:24GiB" \
  --output-dir outputs/ovd_person_offload
```

If you have Flash Attention 2 installed, you can opt in:

```bash
python examples/ovd/infer_ovd.py --attn-implementation flash_attention_2
```
