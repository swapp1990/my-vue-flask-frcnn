<template>
    <div>
        <button :class="getConnectClass()" @click="connectSocket()"><i class="icon-magnet"></i></button>
        <div v-if="connected">
            <div>
                <hr>
                <span v-for="f in imgNames">
                    <button @click="changeSampleImg(f)"> {{f}}</button>
                </span>
                <hr>
            </div>
            <div v-if="!training">
                <hr>
                <div>
                    <div class="p-2">
                        <div class="p-1">
                            <select v-model="selectedLayerIdx">
                                <option disabled value="">Select Layer: </option>
                                <option v-for="l in inception_layers" v-bind:value="l.i">{{l.name}}</option>
                            </select>
                            <span class="ml-3">Selected Layer : {{inception_layers[selectedLayerIdx].name}}</span>
                            <!-- <button @click="modifyFeatureMaps()"><i class="icon-plus"></i></button> -->
                        </div>
                        <input type="range" id="customRange1" min="0" max="512" step="1"
                            v-on:change="changeFMIdx()" v-model="featureMapIdx">
                        <span class="ml-3">Feature Map Index: {{featureMapIdx}}</span>
                    </div>
                    <hr>
                    <span class="p-1"> Feature Viz Details </span>
                    <div class="p-1">
                        <div>
                            <input type="range" id="customRange1" min="32" max="512" step="32"
                            v-on:change="change()" v-model="itern">
                            <span class="ml-3">Iterations: {{itern}}</span>
                        </div>
                        <div>
                            <input type="range" id="customRange1" min="2" max="5" step="1"
                            v-on:change="change()" v-model="pyramidLevels">
                            <span class="ml-3">Laplacian Pyramid Levels: {{pyramidLevels}}</span>
                        </div>
                        <div>
                            <input type="range" id="customRange1" min="0" max="1" step="0.1"
                            v-on:change="change()" v-model="saturation">
                            <span class="ml-3">Saturation: {{saturation}}</span>
                        </div>
                        <div>
                            <input type="range" id="customRange1" min="1" max="64" step="2"
                            v-on:change="change()" v-model="shift">
                            <span class="ml-3">Shift: {{shift}}</span>
                        </div>
                        
                    </div>
                    <hr>
                    <div class="p-3">
                        <button @click="getFeatureMap()">Get Feature Map </button>
                        <button @click="modifyModel()">Modify Model </button>
                    </div>
                    
                </div>
                <hr>
                <div v-if="isLoading" class="spinner-grow" role="status">
                    <span class="sr-only">Loading...</span>
                </div>
                <div v-for="(imgB,i) in imageBytesArr">
                    <button @click="removeFM(i)"><i class="icon-remove"></i></button>
                    <span>{{textArr[i]}}</span>
                    <div v-if="imgB" class="landscape">
                        <img v-bind:src="'data:image/jpeg;base64,'+imgB" />
                    </div>
                </div>
                <hr>
            </div>
        </div>
    </div>
</template>

<script>
import {mapGetters} from 'vuex'
import $ from 'jquery'

import imagesGif from '@/components/ImagesGif.vue';

export default {
    name: "InceptionViz",
    components: {
        imagesGif: imagesGif
    },
    computed: {

    },
    data() {
        return {
            //Socket
            connected: false,
            isLoading: false,
            //Inception
            training: false,
            imgNames: ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
            inception_layers: [
                {i: 0, name:'mixed3a'},
                {i: 1, name:'mixed3b'},
                {i: 2, name:'mixed4a'},
                {i: 3, name:'mixed4b'},
                {i: 4, name:'mixed4c'},
                {i: 5, name:'mixed4d'},
                {i: 6, name:'mixed4e'},
                {i: 7, name:'mixed5a'},
                {i: 8, name:'mixed5b'},
            ],
            selectedLayerIdx: 0,
            featureMapIdx: 0,
            itern: 128,
            pyramidLevels: 4,
            saturation: 0.6,
            shift: 32,
            currId: 0,
            imageBytesArr: [],
            textArr: [],
            imgCache: null,
            imgCacheidx: 0,
            trainLoss: 0,
            batch: 0,
            epoch: 0,
            gifSpeed: 500,
        }
    },
    mounted(){
        this.reset();
        this.connectSocket();
    },
    methods: {
        reset() {
            this.imageBytesArr = [];
            this.textArr = [];
        },
        getConnectClass() {
            let classes = [];
            if(this.connected) {
                classes.push("btn-primary");
            } else {
                classes.push("btn-danger");
            }
            return classes;
        },
        connectSocket() {
            this.socket = io.connect('http://127.0.0.1:5000');
            this.socket.on('connect',()=>{
                console.log("connected");
                this.onConnected();
                this.socket.on('test', (id) => {
                    console.log("test ", id);
                });
                this.socket.on('threadStarted', (id) => {
                    console.log("thread started ", id);
                    this.onThreadStarted(id);
                });
                this.socket.on('General', (msg) => {
                    console.log(msg);
                    this.handleGeneralMsg(msg);
                });

                this.socket.on('workFinished', (obj) => {
                    this.doClientWork(obj);
                });
            });
            this.socket.on('disconnect',()=>{
                console.log('disconnect');
                this.onDisconnected();
            });
            this.socket.on('connect_error', (error) => {
                console.log("Error");
                this.onDisconnected();
            });
            this.socket.on('error', (err) => {
                console.log("Error!", err);
            });

        },
        onConnected() {
            this.socket.emit('init', this.training);
            this.connected = true;
            this.reset();
        },
        onDisconnected() {
            this.socket.close();
            this.connected = false;
        },
        sendMsg() {
            this.socket.emit('sendMsg');
        },
        handleGeneralMsg(msg) {
            if(msg.action) {
                if(msg.action == "layers") {
                    // this.inception_layers = msg.layers;
                } else if(msg.action == "showFig") {
                    this.showFeatureMap(msg.fig)
                }
            }
        },
        changeSampleImg(imgName) {
            this.reset();
            let msg = {'name': imgName};
            if(!this.training) {
                this.socket.emit('changeImg', msg);
            } else {
                this.socket.emit('changeImgTrain', msg);
            }
        },

        onGeneral(msg) {
            let id = msg.id;
            if(msg.action == "load") {
                let selectedThread = this.findActiveThread(id);
                selectedThread.paused = true;
                let content = {id: id, filter_idx: this.currFilterIdx};
                this.socket.emit(msg.action, content);
                this.showArchived = true;
            } else if(msg.action == "filterChange") {
                if(this.showArchived) {
                    console.log(msg.filter_idx);
                    let content = {id: id, filter_idx: msg.filter_idx};
                    this.socket.emit("load", content);
                }
            }
        },

        // Inception Viz Methods
        getFeatureMap() {
            let genDetails = {'itern': this.itern,'pyramidLevels': this.pyramidLevels, 'saturation': this.saturation, 'shift': this.shift};
            let selectedlayer_name = this.inception_layers[this.selectedLayerIdx].name;
            let msg = {'layer_name': selectedlayer_name, 'featureMapIdx': this.featureMapIdx, 'genDetails': genDetails};
            this.socket.emit('getFeatureMap', msg);
            this.isLoading = true;
        },
        modifyModel() {
            this.socket.emit('modifyModel');
        },
        showFeatureMap(f) {
            this.isLoading = false;
            this.imageBytesArr.push(f.axes[0].images[0].data);
            this.textArr.push(f.axes[0].texts[0].text);
        },
        removeFM(i) {
            this.imageBytesArr.splice(i, 1);
            this.textArr.splice(i,1);
        },
        change() {

        }
     }
}
</script>

<style lang="scss" scoped>
.mlp_div {
    background-color: aqua;
    width: 400px;
    height: 400px;
}

$base-spacing-unit: 24px;
$half-spacing-unit: $base-spacing-unit/2;
img {
    max-width: 100%;
    max-height: 100%;
}
.mlp_div {
    background-color: aqua;
}
.header {
	align-items:center;
	background:#FAFAFA;
	padding:$half-spacing-unit;
	border-bottom: 1px solid #eee;
    font-family: 'Dawning of a New Day', cursive;
    font-size: 24px;
    font-weight: bold;
}
</style>