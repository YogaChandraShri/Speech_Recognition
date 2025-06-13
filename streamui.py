# app.py
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa  # To load audio files1
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import time as t
# Set parameters
samplerate = 44100  # Sample rate (Hz)
duration = 10  # Duration of recording in seconds
output_file = "user_audio.wav"  # Output file name
st.title("AI AUDIO")
st.write("PRUVITY HR SOLUTIONS")
st.header("LOGIN")
USER=st.text_input("Enter your name:")
if USER:
    st.write(USER)
    st.write("click the button to start recording")
    if st.button("START"):
        st.write("Recording... Speak into the microphone.")         
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()  # Wait for the recording to finish
        st.write("Recording complete!")
        t.sleep(2)
        # Save the audio to a file
        write(output_file, samplerate, audio)
        st.write("Your text is generating.....")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Load the model
        model_id = "distil-whisper/distil-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        model.to(device)
        # Load the processor
        processor = AutoProcessor.from_pretrained(model_id)

        # Initialize pipeline
        pipe = pipeline(
        "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)
        def load_audio(file_path):
            # Load audio using librosa
            audio, sample_rate = librosa.load(file_path, sr=16000)  # Resample to 16kHz if needed
            return audio

        # Provide path to your own audio file (make sure it's a .wav or other supported format)
        audio_file_path = "user_audio.wav"
        sample = load_audio(audio_file_path)
        # Run the pipeline
        result = pipe(sample)
        t.sleep(5)
        st.write("And your generated text is........")
        st.write(result["text"])  # Output the transcription

