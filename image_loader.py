import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import random
import numpy as np
import matplotlib.pyplot as plt

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_DEPTH = 1
ALLOWED_CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DEFAULT_FONTS = [
    "Roboto-Medium.ttf", 
    "Slabo27px-Regular.ttf", 
    "Montserrat-Regular.ttf", 
    "Merriweather-Regular.ttf"]

WHITE_THRESHOLD=20

def gen_images(sampleCount, saveImages=False, fontList=DEFAULT_FONTS):
    imageResult = np.zeros((sampleCount * len(ALLOWED_CHARS), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    classResult = np.zeros((sampleCount * len(ALLOWED_CHARS), len(ALLOWED_CHARS)))

    for charIndex, ch in enumerate(ALLOWED_CHARS):
        for i in range(0, sampleCount):
            fontSize = random.randint(19, 21)
            fontIndex = random.randint(0, len(fontList) - 1)
            font = ImageFont.truetype(fontList[fontIndex], fontSize)

            img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), (0))
            draw = ImageDraw.Draw(img)

            xPos = random.randrange(-7, IMAGE_WIDTH - 2, 3)
            yPos = random.randrange(-12, IMAGE_HEIGHT - 10, 3)
            draw.text((xPos, yPos), ch, (255), font=font)
            del draw
            
            #Threshold image
            img = img.point(lambda x: 0 if x < WHITE_THRESHOLD else 255, '1')

            if saveImages:
                img.save("image-" + str(charIndex * sampleCount + i) + ".png")
            
            imageData = np.asarray(img)

            #Reshape to have 1 depth.
            imageData = np.reshape(imageData, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

            imageResult[charIndex * sampleCount + i] = imageData

            thisClass = np.zeros(len(ALLOWED_CHARS))
            thisClass[charIndex] = 1.0

            classResult[charIndex * sampleCount + i] = thisClass

    return (imageResult, classResult)

def loadImageData(fileName):
    image = Image.open(fileName)

    #Make sure the image is of right size
    if image.size[0] != IMAGE_WIDTH or image.size[1] != IMAGE_HEIGHT:
        print "Image must be 28x28"
        sys.exit(1)

    #Convert to greyscale and give a depth of 1
    image = image.convert('L')

    #Threshold image
    image = image.point(lambda x: 0 if x < WHITE_THRESHOLD else 255, '1')

    imageData = np.asarray(image)
    #Reshape to have 1 depth.
    imageData = np.reshape(imageData, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    return imageData
    