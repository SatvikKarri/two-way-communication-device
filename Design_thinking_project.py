import google.generativeai as genai
import os
import textwrap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import boto3
import assemblyai as aai
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import joblib
import numpy as np
import time
import RPi.GPIO as GPIO
import board
import busio
import adafruit_mpu6050
import math
from collections import Counter
import simpleaudio as sa
from pydub import AudioSegment

# Configure GenAI API key
genai.configure(api_key='')  # Replace with your actual API key
os.environ['GOOGLE_API_KEY'] = ''  # Replace with your actual API key

# Configure AssemblyAI API key
aai.settings.api_key = ""  # Replace with your actual API key

# Wrap functions in a class structure
class GestureRecognition:
    def __init__(self):
        # Load models and setup
        self.knn = joblib.load('models/KNN')
        self.rf = joblib.load('models/randomforest')
        self.gnb = joblib.load('models/NaiveBayes')
        self.svc = joblib.load('models/SVM')
        self.dtree = joblib.load('models/DecisionTree')
        self.label_encoder = joblib.load('models/label_encoder')
        self.yaw = 0.0
        self.setup_sensors()

    def setup_sensors(self):
        # MPU6050 setup
        i2c = busio.I2C(board.SCL, board.SDA)
        self.mpu = adafruit_mpu6050.MPU6050(i2c)

        # GPIO setup
        GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)
        self.touch_pins = [17, 27, 22, 23, 24]
        for pin in self.touch_pins:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def calculate_roll_pitch_yaw(self, accel_x, accel_y, accel_z, gyro_z, delta_time):
        # Calculate roll, pitch, and yaw as before
        roll = math.atan2(accel_y, accel_z) * (180.0 / math.pi)
        pitch = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2)) * (180.0 / math.pi)
        self.yaw += gyro_z * delta_time
        return roll, pitch, self.yaw

    def read_data(self, delta_time):
        # Read and process sensor data
        accel_x, accel_y, accel_z = self.mpu.acceleration
        gyro_x, gyro_y, gyro_z = [x * 100 for x in self.mpu.gyro]
        roll, pitch, yaw = self.calculate_roll_pitch_yaw(accel_x, accel_y, accel_z, gyro_z, delta_time)
        return roll, pitch, yaw, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

    def read_sensor_data(self, duration=2):
        data = []
        start_time = time.time()

        while time.time() - start_time < duration:
            touch_states = [GPIO.input(pin) for pin in self.touch_pins]
            delta_time = 0.1
            roll, pitch, yaw, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = self.read_data(delta_time)
            data.append([time.time()] + touch_states + [
                accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, roll, pitch, yaw
            ])
            time.sleep(delta_time)

        columns = ['Timestamp', 'Touch1', 'Touch2', 'Touch3', 'Touch4', 'Touch5',
                   'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'Roll', 'Pitch', 'Yaw']
        return pd.DataFrame(data, columns=columns)

    def predict_gesture_over_interval(self, data_window):
        # Prediction with majority voting
        features = data_window[['Touch1', 'Touch2', 'Touch3', 'Touch4', 'Touch5', 'Roll', 'Pitch', 'Yaw', 'AccelX', 'AccelY', 'AccelZ']].mean().to_frame().T
        predictions = {
            "KNN": self.label_encoder.inverse_transform([self.knn.predict(features)[0]])[0],
            "Random Forest": self.label_encoder.inverse_transform([int(round(self.rf.predict(features)[0]))])[0],
            "Naive Bayes": self.label_encoder.inverse_transform([self.gnb.predict(features)[0]])[0],
            "SVM": self.label_encoder.inverse_transform([self.svc.predict(features)[0]])[0],
            "Decision Tree": self.label_encoder.inverse_transform([self.dtree.predict(features)[0]])[0],
        }
        final_prediction = Counter(predictions.values()).most_common(1)[0][0]
        return {"predictions": predictions, "final_prediction": final_prediction}

# Function to print wrapped text
def my_print(output):
    output_str = str(output)
    print(textwrap.fill(output_str, width=120))

# Function to convert text to speech
def text_to_mp3(text, filename):
    session = boto3.Session(
        aws_access_key_id=" ",  # Replace with your actual AWS credentials
        aws_secret_access_key=" ",  # Replace with your actual AWS credentials
        region_name="us-east-1"
    )
    polly = session.client('polly')
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId="Joanna"
    )
    with open(filename, 'wb') as file:
        file.write(response['AudioStream'].read())
    print("mp3 file saved")

    # Play the sound using simpleaudio
    audio = AudioSegment.from_file(filename, format="mp3")
    audio_data = audio.raw_data
    sample_rate = audio.frame_rate
    channels = audio.channels
    bytes_per_sample = audio.sample_width

    play_obj = sa.play_buffer(audio_data, channels, bytes_per_sample, sample_rate)
    play_obj.wait_done()


# Function to record audio
def record_audio(duration, filename="my_recording"):
    # Set the sample rate (samples per second)
    sample_rate = 44100

    # Record the audio
    print("Recording started...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")

    # Get the directory of the script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Save the recorded audio as a WAV file in the same directory as the script
    file_path = os.path.join(script_directory, f"{filename}.wav")
    write(file_path, sample_rate, audio_data)
    print(f"Audio saved as {file_path}")

    # Return the file path
    return file_path

# Function to transcribe audio
def transcribe_audio(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
        return None
    else:
        print(transcript.text)
        return transcript.text

# Function to generate sentence from gesture and text
def generate_sentence(input_text, response_words):
    my_llm = ChatGoogleGenerativeAI(model='gemini-pro')
    my_prompt = PromptTemplate.from_template(
        '''
        You are a sentence artist. Given the input format: input-text: {input_text} response-words: {response_words}
        generate a natural, context-appropriate sentence that incorporates all input words smoothly. Be precise and concise. Keep the sentence focused on the words provided. Respond with only one sentence. 
        When there is a `?` as response-word assume the sentence to be a question.

        For example: 
        input-text: Are you free today? 
        response-words: busy, study, exams
        Output: Sorry, I'm a bit busy with study and prepping for exams, so I might not be free today.

        input-text: Are you today? 
        response-words: Good, you, ?
        Output: I'm doing good and how are you today?
        '''
    )
    chain = LLMChain(
        llm=my_llm,
        prompt=my_prompt,
        verbose=False
    )
    output = chain.run(input_text=input_text, response_words=response_words)
    my_print(output)
    return output

# Main program loop
def main():
    gesture_recognition = GestureRecognition()
    while True:
        # Ask for speaker input (simulated for now)
        speaker_input = input("Is there speaker input? (yes/no): ")

        if speaker_input.lower() == "yes":
            # Capture speech with AssemblyAI
            duration = 5  # Duration in seconds
            audio_file_path = record_audio(duration)
            transcribed_text = transcribe_audio(audio_file_path)

            if transcribed_text:
                # Display transcribed text
                print("Transcribed Text:", transcribed_text)

                # Ask to perform a gesture
                perform_gesture = input("Perform a gesture? (yes/no): ")

                if perform_gesture.lower() == "yes":
                    # Perform gesture and classify
                    data_window = gesture_recognition.read_sensor_data()
                    result = gesture_recognition.predict_gesture_over_interval(data_window)
                    predicted_gesture = result['final_prediction']
                    print("Predicted Gesture:", predicted_gesture)

                    # Generate sentence from transcribed text and predicted gesture
                    sentence = generate_sentence(transcribed_text, predicted_gesture)

                    # Convert text to speech
                    text_to_mp3(sentence, "output.mp3")

        else:
            # Ask to perform a gesture
            perform_gesture = input("Perform a gesture? (yes/no): ")

            if perform_gesture.lower() == "yes":
                # Perform gesture and classify
                data_window = gesture_recognition.read_sensor_data()
                result = gesture_recognition.predict_gesture_over_interval(data_window)
                predicted_gesture = result['final_prediction']
                print("Predicted Gesture:", predicted_gesture)

                # Generate text response
                sentence = generate_sentence("", predicted_gesture)

                # Convert text to speech
                text_to_mp3(sentence, "output.mp3")

if __name__ == "__main__":
    main()