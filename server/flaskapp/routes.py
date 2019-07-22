from app import app

@app.route('/')

@app.route('/login')
def login():
    return 'login'

@app.route('/index')
def index():
    return "Hello, World!"
