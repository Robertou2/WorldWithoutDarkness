import cv2
import os
import pyttsx3
import RPi.GPIO as GPIO
import time
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
from transformers.pipelines import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

# Pin Definition
BUTTON_PIN = 15

# Constants
ASR_MODEL_ID = "openai/whisper-tiny.en"
VISION_MODEL_ID = "unum-cloud/uform-gen2-qwen-500m"
TAKE_PHOTO_COMMAND = "Take a photo."
OUTPUT_DESCRIPTION_FILE = "/home/roberto/Downloads/descriptions.txt"

# Initialize GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('voice', 'english+ml')

# Load models
print('Loading models...')
asr_model = pipeline("automatic-speech-recognition", model=ASR_MODEL_ID, device="cuda")
vision_model = AutoModel.from_pretrained(VISION_MODEL_ID, trust_remote_code=True).to("cuda")
vision_model = torch.compile(vision_model)
processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)


def take_picture(number):
    """
    Capture a picture using the connected camera and save it as a JPEG file.

    Args:
        number (int): The number to be used in the file name.

    Returns:
        str: The file path of the captured image.
    """
    camera_id = "/dev/video0"
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 560)

    if video_capture.isOpened():
        try:
            ret_val, frame = video_capture.read()
            filename = f'/home/roberto/Downloads/prueba_{number}.jpg'
            cv2.imwrite(filename, frame)
            return filename
        finally:
            video_capture.release()
    else:
        print("Unable to open camera")
        return None


def transcribe_mic(chunk_length_s: float) -> str:
    """
    Transcribe audio from the microphone.

    Args:
        chunk_length_s (float): The length of each audio chunk in seconds.

    Returns:
        str: The transcribed text.
    """
    sampling_rate = asr_model.feature_extractor.sampling_rate
    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=chunk_length_s,
    )

    result = ""
    for item in asr_model(mic):
        result = item["text"]
        if not item["partial"][0]:
            break

    return result.strip()


def main():
    photo_number = 1
    filename = take_picture(photo_number)
    image = Image.open(filename)

    try:
        while True:
            button_state = GPIO.input(BUTTON_PIN)
            if button_state == 1:
                print('Waiting for button press...')
            else:
                print("\a")
                question = transcribe_mic(chunk_length_s=5.0)
                print(f"Question: {question}")

                if question == TAKE_PHOTO_COMMAND:
                    print('Taking photo...')
                    photo_number += 1
                    filename = take_picture(photo_number)
                    image = Image.open(filename)
                else:
                    inputs = processor(text=[question], images=[image], return_tensors="pt").to("cuda")
                    print('Generating output...')
                    with torch.inference_mode():
                        output = vision_model.generate(
                            **inputs,
                            do_sample=False,
                            use_cache=True,
                            max_new_tokens=256,
                            eos_token_id=151645,
                            pad_token_id=processor.tokenizer.pad_token_id
                        )

                    prompt_len = inputs["input_ids"].shape[1]
                    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
                    text = decoded_text[:-len("<|im_end|>")]
                    print(f"Response: {text}")

                    with open(OUTPUT_DESCRIPTION_FILE, 'a') as f:
                        f.write(text)
                        f.write('\n')

                    engine.say(text)
                    engine.runAndWait()
                    print("\a")
                    time.sleep(0.5)
                    print("\a")
                    time.sleep(0.5)

    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()