Focused bases blurring using gaussian distribution.
============= 

## Perceptual-focus

Source image.
![Alt text](images/obama.png?raw=true "source image")

Focus blurring.
![Alt text](images/blur.gif?raw=true "blurring based on click points.")

For a give focus point pixel(px,py). Multilple layers are build up were kernel size varies based on normal distribution.
For each layer gausian blur is computed and all the layers are combined in the end.