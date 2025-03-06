import gradio as gr
from PIL import Image, ImageDraw
import cv2, time
from infer_transformers import init_model, run_inference


def visualize_bbox(image, bbox):    
    draw = ImageDraw.Draw(image) 
    draw.rectangle(list(map(int, bbox)), outline=(255, 0, 0) , width=2)
    return image
    
def main_process(image, description):

    start_time = time.time()
    out = run_inference(model, processor, image, description)
    print("⚡️⚡️⚡️ Formatted Output: ",out)
    print("⚡️⚡️⚡️ Inference Time: ", time.time()-start_time)
    
    return out['think'], visualize_bbox(image, out['bbox_2d']) 


if __name__ == "__main__":
    example_pairs = [
        ["</your/path/to//demo/images/image1.jpg>", "person with blue shirt"],
        ["</your/path/to//demo/images/image2.jpg>", "food with the highest protein"],
        ["</your/path/to//demo/images/image3.jpg>", "the cheapest Apple laptop"],
    ]
    
    model_path_dir = "</your/path/to/Qwen2.5-VL-3B-VLM-R1-REC-500steps>"
    model, processor = init_model(model_path_dir)
    
    iface = gr.Interface(
        fn=main_process,  
        inputs=[gr.Image(type="pil", label="Input Image"), gr.Textbox(label="Description Text")], 
        outputs=[gr.Textbox(label='Thinking Process'), gr.Image(type="pil", label="Result with Bbox")], 
        live=False,  
        examples=example_pairs,
        description="Upload an image and input description text, the system will return the thinking process and region annotation.",
        article="Project's GitHub: [VLM-R1](https://github.com/VLM-R1)"
    )
    iface.launch()