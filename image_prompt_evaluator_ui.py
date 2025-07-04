import gradio as gr
from image_prompt_evaluator import ImagePromptEvaluator
import os

# Initialize the evaluator
evaluator = ImagePromptEvaluator()

def evaluate_image_and_prompt(image, prompt, threshold=0.25):
    """Evaluate the similarity between an image and a prompt"""
    # Save the image temporarily
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # Evaluate the image
    results = evaluator.evaluate_image(temp_path, prompt, threshold)
    
    # Format the results
    output = f"## Image-Prompt Evaluation Results\n\n"
    output += f"**Prompt:** \"{results['prompt']}\"\n\n"
    output += f"**Overall Match:** {results['percentage_match']} ({results['quality']})\n\n"
    output += f"**Feedback:** {results['feedback']}\n\n"
    
    output += "### Keyword Analysis:\n"
    for kw in results['keyword_analysis']:
        status = "✓" if kw['present'] else "✗"
        output += f"- {status} {kw['keyword']} - {kw['confidence']}\n"
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return output

# Create the Gradio interface
with gr.Blocks(title="DF-GAN Image-Prompt Evaluator") as demo:
    gr.Markdown("# DF-GAN Image-Prompt Accuracy Evaluator")
    gr.Markdown("Upload a generated image and enter the prompt used to generate it to evaluate how well they match.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Generated Image")
            prompt_input = gr.Textbox(label="Text Prompt", placeholder="Enter the prompt used to generate the image")
            threshold_input = gr.Slider(minimum=0.1, maximum=0.4, value=0.25, step=0.01, label="Similarity Threshold")
            evaluate_button = gr.Button("Evaluate")
        
        with gr.Column():
            result_output = gr.Markdown(label="Evaluation Results")
    
    evaluate_button.click(
        fn=evaluate_image_and_prompt,
        inputs=[image_input, prompt_input, threshold_input],
        outputs=result_output
    )

if __name__ == "__main__":
    demo.launch()
