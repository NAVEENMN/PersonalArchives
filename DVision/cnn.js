// Require the client
var Clarifai = require('clarifai');

// instantiate a new Clarifai app passing in your clientId and clientSecret
var app = new Clarifai.App(
  'vfDCvKs3-PJOciwXB6DS6V7nCPPvoMRJYZkj6i6p',
  'AP1nAt40RITePTocRYX-MGDxmVOke-cMZUbLwq7r'
);

app.models.predict(Clarifai.GENERAL_MODEL, 'https://samples.clarifai.com/metro-north.jpg').then(
  function(response) {
    var obj = JSON.parse(response);
    console.log(obj)
  },
  function(err) {
    console.error(err);
  }
);
