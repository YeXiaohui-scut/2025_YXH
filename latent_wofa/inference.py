from my_model_library import WatermarkedStableDiffusionPipeline

def load_models():
    # Load pre-trained models
    model = WatermarkedStableDiffusionPipeline()
    return model

def generate_images(model, inputs):
    with torch.no_grad():
        watermarked_images = model.generate(inputs)
    return watermarked_images

def extract_watermarks(model, images):
    with torch.no_grad():
        extracted_watermarks = model.extract_watermarks(images)
    return extracted_watermarks

if __name__ == '__main__':
    model = load_models()
    # Assume 'input_images' is pre-loaded
    watermarked_images = generate_images(model, input_images)
    extracted_watermarks = extract_watermarks(model, watermarked_images)