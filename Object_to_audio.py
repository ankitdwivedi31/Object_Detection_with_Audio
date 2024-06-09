

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import scipy.io.wavfile as wavfile
import numpy as np

#object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
#Narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
model_path = "C:/Users/ankitdwivedi/OneDrive - Adobe/Desktop/NLP Projects/Video to Text Summarization/Model/models--facebook--detr-resnet-50/snapshots/1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"

tts_model_path = "C:/Users/ankitdwivedi/OneDrive - Adobe/Desktop/NLP Projects/Video to Text Summarization/Model/models--kakao-enterprise--vits-ljs/snapshots/3bcb8321394f671bd948ebf0d086d694dda95464"

object_detector = pipeline("object-detection", model=model_path)

Narrator = pipeline("text-to-speech", model=tts_model_path)

##read the image file
# raw_image =Image.open("C:/Users/ankitdwivedi/OneDrive - Adobe/Desktop/NLP Projects/Video to Text Summarization/Image_Processing/918oQOaXZTL._AC_UF1000,1000_QL80_.jpg")
# output = object_detector(raw_image)
#print(output)

# def generate_audio(text):
#     Narrated_Text = Narrator(text)
#     wavfile.write("fine_tuned_audio.wav", rate = Narrated_Text["sampling_rate"], data = Narrated_Text["audio"][0])
#     return "output.wav"

def generate_audio(text):
    Narrated_Text = Narrator(text)
    audio_data = np.array(Narrated_Text["audio"][0])
    sampling_rate = Narrated_Text["sampling_rate"]
    wavfile.write("generated_audio.wav", rate=sampling_rate, data=audio_data)
    return "generated_audio.wav"

def read_objects(detection_objects):
    # Initialize counters for each object label
    object_counts = {}

    # Count the occurrences of each label
    for detection in detection_objects:
        label = detection['label']
        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1

    # Generate the response string
    response = "This picture contains"
    labels = list(object_counts.keys())
    for i, label in enumerate(labels):
        response += f" {object_counts[label]} {label}"
        if object_counts[label] > 1:
            response += "s"
        if i < len(labels) - 2:
            response += ","
        elif i == len(labels) - 2:
            response += " and"

    response += "."

    return response

def draw_bounding_boxes(image, detections, font_path=None, font_size=20):
    """
    Draws bounding boxes on the given image based on the detections.
    :param image: PIL.Image object
    :param detections: List of detection results, where each result is a dictionary containing
                       'score', 'label', and 'box' keys. 'box' itself is a dictionary with 'xmin',
                       'ymin', 'xmax', 'ymax'.
    :param font_path: Path to the TrueType font file to use for text.
    :param font_size: Size of the font to use for text.
    :return: PIL.Image object with bounding boxes drawn.
    """
    # Make a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Load custom font or default font if path not provided
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        # When font_path is not provided, load default font but it's size is fixed
        font = ImageFont.load_default()
        # Increase font size workaround by using a TTF font file, if needed, can download and specify the path

    for detection in detections:
        box = detection['box']
        xmin = box['xmin']
        ymin = box['ymin']
        xmax = box['xmax']
        ymax = box['ymax']

        # Draw the bounding box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        # Optionally, you can also draw the label and score
        label = detection['label']
        score = detection['score']
        text = f"{label} {score:.2f}"

        # Draw text with background rectangle for visibility
        if font_path:  # Use the custom font with increased size
            text_size = draw.textbbox((xmin, ymin), text, font=font)
        else:
            # Calculate text size using the default font
            text_size = draw.textbbox((xmin, ymin), text)

        draw.rectangle([(text_size[0], text_size[1]), (text_size[2], text_size[3])], fill="red")
        draw.text((xmin, ymin), text, fill="white", font=font)

    return draw_image

def detect_object(image):
    raw_image = image
    output = object_detector(raw_image)
    processed_image = draw_bounding_boxes(raw_image, output)
    natural_text = read_objects(output)
    processed_audio = generate_audio(natural_text)

    return processed_image, processed_audio

demo = gr.Interface(fn=detect_object,
                    inputs=[gr.Image(label="Select Image",type="pil")],
                    outputs=[gr.Image(label="Processed Image", type="pil"), 
                             gr.Audio(label="Generated_Audio")],
                    title="Project 7: Object Detector with Audio",
                    description="THIS APPLICATION WILL BE USED TO DETECT OBJECTS and Audio for objects mentioned INSIDE THE PROVIDED INPUT IMAGE.")
demo.launch()
