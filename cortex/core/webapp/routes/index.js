var express = require('express');
var router = express.Router();

const name = 'Naveen'

mapboxgl.accessToken = 'pk.eyJ1IjoibXlzb3JuMSIsImEiOiIwODY0MDQ0ZTdiYTViYmU0ZTZiOGI4OTU5MjQxZGY1NCJ9.elqMFi4oQtFFjYXbW2Oxig';
var beforeMap = new mapboxgl.Map({
container: 'before',
style: 'mapbox://styles/mapbox/satellite-v9',
center: [-74.50, 40],
zoom: 9
});
var afterMap = new mapboxgl.Map({
container: 'after',
style: 'mapbox://styles/mapbox/streets-v11',
center: [-74.50, 40],
zoom: 9
});
var map = new mapboxgl.Compare(beforeMap, afterMap, {
// Set this to enable comparing two maps by mouse movement:
// mousemove: true
});
var draw = new MapboxDraw({
displayControlsDefault: false,
controls: {
polygon: true,
trash: true
}
});
afterMap.addControl(draw);
afterMap.on('draw.create', updateArea);
afterMap.on('draw.delete', updateArea);
afterMap.on('draw.update', updateArea);
function updateArea(e) {
var data = draw.getAll();
var answer = document.getElementById('calculated-area');
if (data.features.length > 0) {
var area = turf.area(data);
// restrict to area to 2 decimal points
var rounded_area = Math.round(area*100)/100;
answer.innerHTML = '<p><strong>' + rounded_area + '</strong></p><p>square meters</p>';
} else {
answer.innerHTML = '';
if (e.type !== 'draw.delete') alert("Use the draw tools to draw a polygon!");
}
}

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index.ejs', {title: 'Hello', name:name});
});

module.exports = router;
