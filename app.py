import gradio as gr
from transformers import pipeline
import whisper
import cv2
import numpy as np
import mediapipe as mp
import torch

# Load Whisper model for voice transcription
device = "cuda" if torch.cuda.is_available() else "cpu"
transcribe_model = whisper.load_model("base", device=device)

# Load a pre-trained language model for skincare advice
skincare_model = pipeline("text-generation", model="distilgpt2", device=0 if device == "cuda" else -1)

# Initialize MediaPipe for facial landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def apply_virtual_makeup(image):
    # Detect facial landmarks and apply virtual makeup
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks with makeup effect
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)  # Apply "blush" with blue points as example

    return image

def transcribe_voice(audio):
    transcription = transcribe_model.transcribe(audio)["text"]
    return transcription

def generate_skincare_recommendation(text_input):
    prompt = f"Generate skincare advice for the following concerns: {text_input}"
    response = skincare_model(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Gradio Interface
def skincare_consultant(input_image, audio_input=None):
    if audio_input:
        question_text = transcribe_voice(audio_input)
    else:
        question_text = "Give general skincare advice."

    skincare_advice = generate_skincare_recommendation(question_text)
    makeup_applied_image = apply_virtual_makeup(input_image)

    return makeup_applied_image, skincare_advice

# Set up Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AR-Based AI Makeup and Skincare Consultant")

    with gr.Row():
        input_image = gr.Image(type="numpy", label="Upload Face Image")  # Changed to upload instead of webcam capture
        audio_input = gr.Audio(type="filepath", label="Ask a Skincare Question (Optional)")

    makeup_image = gr.Image(label="Virtual Makeup Applied")
    skincare_advice_output = gr.Textbox(label="Skincare Advice")

    submit_button = gr.Button("Get Advice and Apply Makeup")
    submit_button.click(
        skincare_consultant,
        inputs=[input_image, audio_input],
        outputs=[makeup_image, skincare_advice_output]
    )

demo.launch()
