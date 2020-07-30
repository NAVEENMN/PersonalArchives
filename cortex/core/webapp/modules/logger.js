const EventEmitter = require('events');

class Logg extends EventEmitter {
  log(message) {
    // log the message
    console.log(message);
    // raise an event
    this.emit('messageLogged', message);
  }
}

module.exports = Logg;
