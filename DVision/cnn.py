from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from clarifai import rest
from clarifai.rest import ClarifaiApp

app = ClarifaiApp("vfDCvKs3-PJOciwXB6DS6V7nCPPvoMRJYZkj6i6p", "AP1nAt40RITePTocRYX-MGDxmVOke-cMZUbLwq7r")

# get the general model
model = app.models.get("general-v1.3")

# predict with the model
image = ClImage(url='https://samples.clarifai.com/metro-north.jpg')
model.predict_by_url([image])

print model
