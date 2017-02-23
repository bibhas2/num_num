This is a Convolutional Neural Net to recognize alphanumeric text of different fonts and sizes. 
There are 36 characters to recognize (all letters are in uppercase).

During training and validation images are randomly varied using:

- Different fonts
- Different font sizes
- Different x and y position for the letters

## Training
Training takes about 5 minutes in a laptop. Run training:

```
python train.py
```

This will save the weights nd biases in a file.

## Cross Validation
Two sets of validations are run. The first one uses known fonts used in training. The second pass uses unseen fonts.

Run validation:

```
python validate.py
```

## Prediction
You can create your own image. It must be a 28x28 PNG with the letter in white against a black background. 
Image of any depth (RGB or RGBA) are allowed.

```
python predict.py file.png
```

## Observations
We use a very simple CNN. It has three convolutaional layers:

1. Height: 5, Width: 5, depth: 4
2. Height: 5, Width: 5, depth: 8
3. Height: 5, Width: 5, depth: 12

There is a fully connected layer with 400 neurons.

Accuracy during cross validation is excellent for both seen and unseen fonts.

For some reason prediction using hand made images does not seem that great. The model seems to be highly susceptible 
to the letter's x and y position.
