import logging
import threading
import time
import json
import concurrent.futures

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit
import eventlet
eventlet.monkey_patch()

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}}) 
#executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
thread_run = {}
# def thread_function(name):
#     logging.info("Thread %s: starting", name)
#     time.sleep(5)
#     logging.info("Thread %s: finishing", name)
#     emit('threadFinished', name)
def emitFinished(step):
    #socketio.emit('threadFinished', dict(foo='bar'))
    socketio.emit('threadFinished', step)

def listen(step):
    logging.info("Thread %s: starting", step)
    eventlet.sleep(5)
    emitFinished(step)
    logging.info("Thread %s: finishing", step)

@app.route('/startThread', methods=['POST'])
def startThread():
    print("start thread")
    data = json.loads(request.data)
    start = data['name']
    #executor.submit(thread_function, start)
    return "test"

@socketio.on('startNewThread')
def startNewThread(step):
    #executor.submit(thread_function, step)
    eventlet.spawn(listen, step)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    print("running socketio")
    socketio.run(app)
    #app.run(debug=True, use_reloader=False)