import speech_recognition as sr
from transformers import pipeline

recognizer = sr.Recognizer()
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Specify the model explicitly
nlp_model = pipeline("sentiment-analysis", model=model_name)

# Capture audio from the microphone
with sr.Microphone() as source:
    print("Listening...")
    audio = recognizer.listen(source)

# Recognize speech and process with NLP model
try:
    print("Recognizing...")
    text = recognizer.recognize_google(audio)
    print(f"Recognized Text: {text}")
    
    # Apply NLP model to the recognized text
    result = nlp_model(text)
    print(f"Sentiment: {result}")
except sr.UnknownValueError:
    print("Could not understand the audio")
except sr.RequestError as e:
    print(f"Request error from Google Speech Recognition service: {e}")
