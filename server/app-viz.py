from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response

from keras.applications import VGG16
from keras import activations
from util import viz_utils
from vis import activation_maximization as am
from matplotlib import pyplot as plt

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

def init():
    model = VGG16(weights='imagenet', include_top=True)
    layer_idx = viz_utils.find_layer_idx(model, 'predictions')
    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = viz_utils.apply_modifications(model)
    img = am.visualize_activation(model, layer_idx, filter_indices=20)
    plt.imshow(img)
    plt.show()
init()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)