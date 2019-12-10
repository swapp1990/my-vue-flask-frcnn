<template>
    <div>
        <button :class="getConnectClass()" @click="connectSocket()"><i class="icon-magnet"></i></button>
        <div v-if="connected">
            <hr>
            <div>
                <span v-for="f in imgNames">
                    <button @click="changeSampleImg(f)"> {{f}}</button>
                </span>
            </div>
            <div>
                <button @click="getAllActivations()"> All Activations</button>
            </div>
            <div>
                <select v-model="selectedLayer">
                    <option disabled value="">Select Layer: </option>
                    <option v-for="l in inception_layers" v-bind:value="l.i">{{l.name}}</option>
                </select>
                <button @click="getFeatureMaps()"><i class="icon-plus"></i></button>
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
</template>

<script>
import {mapGetters} from 'vuex'
import $ from 'jquery'

export default {
    name: "InceptionTrain",
    components: {
    },
    computed: {

    },
    data() {
        return {
            //Socket
            connected: false,
            isLoading: false,
            //Inception
            imgNames: ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
            inception_layers: [],
            currId: 0,
            selectedLayer: null,
            imageBytesArr: [],
            textArr: []
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
                this.socket.on('gotfig', (fig) => {
                    this.displayImg(fig);
                });
                this.socket.on('gotfigs', (figs) => {
                    this.displayFigs(figs);
                });
                this.socket.on('layer_names', (arr) => {
                    this.gotLayerNames(arr);
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

        },
        onConnected() {
            this.socket.emit('init');
            this.connected = true;
            this.reset();
        },
        onDisconnected() {
            this.socket.close();
            this.connected = false;
        },
        gotLayerNames(arr) {
            this.inception_layers = arr;
        },
        changeSampleImg(imgName) {
            this.reset();
            let msg = {'name': imgName};
            this.socket.emit('changeImg', msg);
        },
        displayImg(content) {
            let mlpId = '#mlp_fig_1';
            var graph = $(mlpId);
            // console.log(content);
            this.imageBytesArr.push(content.axes[0].images[0].data);
            this.textArr.push(content.axes[0].texts[0].text);
        },
        displayFigs(figs) {
            // console.log(figs);
            figs.forEach(f => {
                this.imageBytesArr.push(f.axes[0].images[0].data);
                this.textArr.push(f.axes[0].texts[0].text);
            });
            this.isLoading = false;
        },
        getAllActivations() {
            this.imageBytesArr = [];
            this.textArr = [];
            this.isLoading = true;
            this.socket.emit('filteredFm', {filter: 'activations'});
        },
        getFeatureMaps() {
            console.log(this.selectedLayer);
            if(this.selectedLayer != null) {
                this.socket.emit('allFm', this.selectedLayer);
            }
        },
        removeFM(i) {
            this.imageBytesArr.splice(i, 1);
            this.textArr.splice(i,1);
        },
        addNewThread() {
            if(!this.layers_map[this.selectedLayer]) {
                this.layers_map[this.selectedLayer] = {currFilter: 0};
            } else {
                this.layers_map[this.selectedLayer].currFilter++;
            }
            console.log(this.layers_map);
            this.activeThreads.push({id: this.currId, step: 0, layer: this.selectedLayer, 
                        filter_idx: this.layers_map[this.selectedLayer].currFilter});
            
            this.currId++;
        },
        initThreads() {
            //Show normal viz
            let id=0;
            this.activeThreads.push({id: 0, step: 0});
        },
        doClientWork(obj) {
            let imgData = obj.fig;
            let selectedThread = this.findActiveThread(obj.id);
            selectedThread.step += 1;
            selectedThread.figImg = imgData;
            setTimeout(() => {
                this.socket.emit('perform', obj.id);
            }, 100);
            if(selectedThread.step % 50 == 0) {
                this.socket.emit('save', obj.id);
                console.log("Image saved");
            }
        },
        startNewThread(panelParam) {
            console.log(panelParam);
            let style = {height: this.figHeight, width: this.figWidth};
            panelParam.style = style;
            this.socket.emit('startNewThread', panelParam);
        },
        //Socket events
        onThreadStarted(id) {
            console.log("onThreadStarted ", id);
            this.socket.emit('perform', id);
        },
        //Events
        onCreateThread(panelParam) {
            this.startNewThread(panelParam);
        },
        onPerform(msg) {
            let selectedThread = this.findActiveThread(msg.id);
            if(this.currFilterIdx !== msg.f) {
                console.log("Filter index changed. Modify");
                this.currFilterIdx = msg.filter_idx;
                let selectedThread = this.findActiveThread(msg.id);
                selectedThread.step = 0;
                this.socket.emit('modify', msg.panelParam);
                selectedThread.paused = false;
            } else {
                selectedThread.paused = false;
                this.socket.emit('perform', msg.id);
            }
        },
        onPause(id) {
            let selectedThread = this.findActiveThread(id);
            selectedThread.paused = true;
        },
        onStop(id) {
            this.socket.emit('stopThread', id);
        },
        onSave(id) {
            let selectedThread = this.findActiveThread(id);
            selectedThread.paused = true;
            this.socket.emit('save', id);
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
        //Thread methods
        handleWorkException(id) {
            let selectedThread = this.findActiveThread(id);
            // selectedThread.paused = true;
            console.log(this.activeThreads[0]);
            this.activeThreads[0] = {id: selectedThread.id, figImg: selectedThread.figImg, step: selectedThread.step, paused: true};
            console.log(this.activeThreads);
        },
        findActiveThread(id) {
            var index = this.activeThreads.map(function(e) { return e.id; }).indexOf(id);
            if(index != -1)
                return this.activeThreads[id];
            else 
                return undefined
        },
        removeActiveThread(id) {
            var index = this.activeThreads.map(function(e) { return e.id; }).indexOf(id);
            if (index !== -1) this.activeThreads.splice(index, 1);
        },
    }
}
</script>

<style lang="scss" scoped>
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