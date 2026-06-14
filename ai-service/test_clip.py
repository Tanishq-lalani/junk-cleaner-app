import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def main():
    # 1. Explicitly force PyTorch to use the CPU
    device = "cpu"
    print(f"Using device: {device.upper()}")

    # 2. Load the model and processor 
    print("Loading CLIP model from Hugging Face... (This might take a minute initially)")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(" Model loaded successfully!")

    # 3. Define the path to your test image
    image_path = "sample3.jpeg"
    
    if not os.path.exists(image_path):
        print(f" Error: Please place a test image named '{image_path}' inside your ai-service folder.")
        return

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f" Error opening image: {e}")
        return

    # 4. Define categories we want to test the image against
    labels = [
        "a family or vacation photo", 
        "an internet meme or funny image with text edited on it", 
        "a document, invoice, receipt or bill", 
        "a screenshot of text or chat conversations",
        "a male photo",
        "a female photo",
    ]

    print(f" Analyzing '{image_path}' against categories...")

    # 5. Process image and text, then pass through the model
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad(): # Tells PyTorch not to calculate gradients
        outputs = model(**inputs)
    
    # 6. Calculate percentages/probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1)
    
    # 7. Print the results clearly
    print("\n --- Classification Results ---")
    for label, prob in zip(labels, probs[0]):
        print(f"🔹 {label}: {prob.item() * 100:.2f}%")

if __name__ == "__main__":
    main()