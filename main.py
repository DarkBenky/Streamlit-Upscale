import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the upscaling model
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def upscale_image(model, image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Check if the image is RGBA, convert to RGB if it is
    if img_array.shape[-1] == 4:
        img_array = img_array[:,:,:3]
    
    # Ensure the image has 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    
    # Normalize the image
    input_image = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    
    # Perform upscaling
    upscaled = model.predict(input_image)
    
    # Remove batch dimension and denormalize
    upscaled = np.squeeze(upscaled, axis=0)
    upscaled = np.clip(upscaled * 255, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    return Image.fromarray(upscaled)

def main():
    st.title("Image Upscaling App")

    # Load the model
    model_path = "upscaling_model_multi_dataset_4x.h5"
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        # Upscale button
        if st.button("Upscale Image"):
            with st.spinner("Upscaling..."):
                try:
                    # Perform upscaling
                    upscaled_image = upscale_image(model, image)
                    
                    # Display upscaled image
                    st.image(upscaled_image, caption="Upscaled Image", use_column_width=True)
                    
                    # Provide download link for upscaled image
                    buf = io.BytesIO()
                    upscaled_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download Upscaled Image",
                        data=byte_im,
                        file_name="upscaled_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"An error occurred during upscaling: {str(e)}")

if __name__ == "__main__":
    main()