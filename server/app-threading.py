import logging
import threading
import time
import json
import concurrent.futures
import queue

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit
import eventlet
eventlet.monkey_patch()

import global_vars as G

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

def wait_for_event(e):
    print('wait for event')
    time.sleep(2)
    socketio.emit('threadFinished', 0)

    print(e)
    event_is_set = e.wait()
    print('event_is_set ', event_is_set)

active_queues = []
class Worker(threading.Thread):
    def __init__(self, id):
        threading.Thread.__init__(self)
        self.mailbox = queue.Queue()
        self.id = id
        self.step = 0
        active_queues.append(self.mailbox)
                    
    def doWork(self):
        self.step = self.step+1
        time.sleep(2)
        obj = {"id": self.id, "step": self.step}
        socketio.emit('workFinished', obj)

    def run(self):
        while True:
            data = self.mailbox.get()
            if 'action' in data:
                if data['action'] == 'stop':
                    print(self, 'shutting down')
                    if self.id == data['id']:
                        socketio.emit('threadStopped', self.id)
                        return
            print(self, 'received a message:', data['id'])
            if self.id == data['id']:
                self.doWork()
    
    def getId(self):
        return self.id

    def stop(self):
        print("stop thread", self.id)
        # socketio.emit('threadFinished', 0)
        active_queues.remove(self.mailbox)
        self.mailbox.put("shutdown")
        self.join()

def broadcast_event(data):
    for q in active_queues:
        q.put(data)

@socketio.on('startNewThread')
def startNewThread(id):
    thread = Worker(id)
    thread.start()
    print("thread " + id + " started")
    G.active_threads.append(thread)
    # eventlet.spawn(listen, step)
    # G.thread_e = threading.Event()
    # t1 = threading.Thread(name='blocking', 
    #                   target=wait_for_event,
    #                   args=(G.thread_e,))
    # t1.start()
@socketio.on('stopThread')
def stopThread(id):
    msg = {"action": "stop", "id": id}
    broadcast_event(msg)
    for act_t in G.active_threads:
        if act_t.getId() == id:
            act_t.stop()
            G.active_threads.remove(act_t)

@socketio.on('incrementStep')
def incrementStep(id):
    print(id)
    msg = {"id": id}
    #G.thread_e.set(stateToRemov)
    broadcast_event(msg)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    print("running socketio")
    socketio.run(app)
    #app.run(debug=True, use_reloader=False)