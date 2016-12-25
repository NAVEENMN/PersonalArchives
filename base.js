var express = require('express');
var arDrone = require('ar-drone');
var fs = require('fs');
var http = require('http');

//require('ar-drone-png-stream')(client, { port: 8080 });
var client  = arDrone.createClient();
//var app = express();
//console.log(__dirname);
//app.use(express.static(__dirname + '/public'));
//app.listen(8000);
var pngStream = client.getPngStream();
var frameCounter = 0;
var period = 5000; // Save a frame every 5000 ms.
var lastFrameTime = 0;

var lastPng;

pngStream
  .on('error', console.log)
  .on('data', function(pngBuffer) {
    var now = (new Date()).getTime();
    if (now - lastFrameTime > period) {
      frameCounter++;
      lastFrameTime = now;
      console.log('Saving frame');
	lastPng = pngBuffer;
	/*
      fs.writeFile('frame.png', pngBuffer, function(err) {
        if (err) {
          console.log('Error saving PNG: ' + err);
        }
      });
	*/
	//app.use('/static', express.static('public'))
    }
  });

var server = http.createServer(function(req, res) {
if (!lastPng) {
  res.writeHead(503);
  res.end('Did not receive any png data yet.');
  return;
}

res.writeHead(200, {'Content-Type': 'image/png'});
res.end(lastPng);
});

server.listen(8080, function() {
   console.log('Serving latest png on port 8080 ...');
})
