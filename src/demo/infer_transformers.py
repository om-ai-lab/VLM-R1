import os, torch, re, json, random, argparse, io, base64

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def image2base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    
def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            x1, y1, x2, y2 = bbox
            return bbox, False
    return [0, 0, 0, 0], False

    
def init_model(model_path_dir):
    random.seed(42)
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    # default processer
    processor = AutoProcessor.from_pretrained(model_path_dir)

    return model, processor
    

def gen_message_s(image_or_path, describes):
    QUESTION_TEMPLATE = "Please provide the bounding box coordinate of the region this sentence describes: {Question}. First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    
    if isinstance(image_or_path, str):
        img = f"file://{image_or_path}"
    else:
        img = f"data:image/png;base64,{image2base64(image_or_path)}"
    
    message = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": img
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=describes)
                }
            ]
        }]
    return [message]
 
def extract_answer(input_str):
    
    think_pattern = r'<think>(.*?)</think>'
    think_content = re.search(think_pattern, input_str, re.DOTALL)
    think_text = think_content.group(1).strip()
    
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_content = re.search(answer_pattern, input_str)
    answer_text = answer_content.group(1).strip()
    
    bbox = extract_bbox_answer(input_str)[0]
        
    return {'think': think_text,
            'answer': answer_text, 
            'bbox_2d': bbox}

def run_inference(model, processor, image_path, describes):
    batch_messages = gen_message_s(image_path, describes)
    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, use_fast=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, 
                                   ) # do_sample=False # remove warnings.warn
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("\n ⚡️⚡️⚡️ Orignal Output: \n", batch_output_text[0])
    return extract_answer(batch_output_text[0])
    
        
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with VLM-R1")
    
    parser.add_argument('--model_path', '-m', type=str,
                        default="</your/path/to/Qwen2.5-VL-3B-VLM-R1-REC-500steps>", 
                        help="Path to the model checkpoint")
    parser.add_argument('--image_path', '-i', type=str,
                        default="</your/path/to/demo/images/image1.jpg>",
                        help="Path to the image for inference")
    parser.add_argument('--describes', '-d', type=str,
                        default="<your_description.>", 
                        help="Description or caption for the image")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, processor = init_model(args.model_path)
    out = run_inference(model, processor, args.image_path, args.describes)
    print("⚡️"*20, "\n", out, "\n", "⚡️"*20)
    
    
    

    
        
 