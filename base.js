var arDrone = require('ar-drone');
var fs = require('fs');
var http = require('http');

var client  = arDrone.createClient();
var pngStream = client.getPngStream();
var frameCounter = 0;
var period = 100; // Save a frame every 5000 ms.
var lastFrameTime = 0;

var lastPng;

pngStream
  .on('error', console.log)
  .on('data', function(pngBuffer) {
	lastPng = pngBuffer;
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
