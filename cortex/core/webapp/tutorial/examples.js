const EventEmitter = require('events');
const emitter = new EventEmitter();

// Register a listener
emitter.on('messageLogged', function(arg){
  console.log('triggered..', arg);
});

// other way of writing
/*
emitter.on('messageLogged', (arg) => {
  console.log('triggered..', arg);
});
*/

// Raise an event
// listener must be first setup.
const data = {id: 1, url: 'http://'}
emitter.emit('messageLogged', data);

// inside class not need to mention function keyword
class Logger {
  log(message) {
    // log the message
    console.log(message);
    // raise an event
    emitter.emit('messageLogged', data);
  }
}

// when you extend you can use this to refer internal elements
class Logger extends EventEmitter {
  log(message) {
    // log the message
    console.log(message);
    // raise an event
    this.emit('messageLogged', data);
  }
}
