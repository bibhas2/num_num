import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import random
import numpy as np
import matplotlib.pyplot as plt

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_DEPTH = 3
ALLOWED_CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def gen_images(sampleCount, saveImages=False):
    fontList = [
        "Roboto-Medium.ttf", 
        "Slabo27px-Regular.ttf", 
        "Montserrat-Regular.ttf", 
        "Merriweather-Regular.ttf"]

    imageResult = np.zeros((sampleCount * len(ALLOWED_CHARS), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    classResult = np.zeros((sampleCount * len(ALLOWED_CHARS), len(ALLOWED_CHARS)))

    for charIndex, ch in enumerate(ALLOWED_CHARS):
        for i in range(0, sampleCount):
            fontSize = random.randint(19, 21)
            fontIndex = random.randint(0, len(fontList) - 1)
            font = ImageFont.truetype(fontList[fontIndex], fontSize)

            img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (0,0,0))
            draw = ImageDraw.Draw(img)

            xPos = random.randint(2, 10)
            yPos = random.randint(2, 10)
            draw.text((xPos, yPos), ch, (255, 255, 255), font=font)
            del draw
            
            if saveImages:
                img.save("image-" + str(charIndex * sampleCount + i) + ".png")

            imageResult[charIndex * sampleCount + i] = np.asarray(img) 

            thisClass = np.zeros(len(ALLOWED_CHARS))
            thisClass[charIndex] = 1.0

            classResult[charIndex * sampleCount + i] = thisClass

    return (imageResult, classResult)
