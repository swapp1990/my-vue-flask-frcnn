<template>
    <div>
        <button @click="initThreads()">Start Threads</button>
        <div v-for="t in activeThreads" class="col-sm">
            <viz-panel :t="t"
                        v-on:createThread="onCreateThread"
                        v-on:perform="onPerform"
                        v-on:pause="onPause"
                        v-on:stop="onStop"
                        v-on:save="onSave"
                        v-on:general="onGeneral"></viz-panel>
        </div>
    </div>
</template>

<script>
import {mapGetters} from 'vuex'
import $ from 'jquery'
import spinner from '@/components/Spinner'
import VizPanel from '@/components/VizPanel'

export default {
    name: "SingleFeat",
    components: {
        spinner: spinner,
        VizPanel: VizPanel
    },
    computed: {
        
    },
    data() {
        return {
            socket: null,
            step: 0,
            stopped: false,
            activeThreads: [],
            stepLimit: 2005,
            figHeight: 300,
            figWidth: 300,
            currFilterIdx: -1,
            showNegative: false,
            showArchived: false
        }
    },
    mounted(){
        this.socket = io.connect('http://127.0.0.1:5000');
        this.socket.on('connect',()=>{
            console.log("connected");
            this.socket.on('test', (id) => {
                console.log("test ", id);
            });
            this.socket.on('threadStarted', (id) => {
                console.log("thread started");
                this.socket.emit('perform', id);
            });
            this.socket.on('threadFinished', (id) => {
                console.log('threadFinished', id);
                this.removeActiveThread(id);
            });
            this.socket.on('workFinished', (obj) => {
                // console.log('workFinished', obj);
                if(obj.notFound) {
                    console.log("Not Found");
                } else if(obj.exception) {
                    //General Exception returned to Thread to handle what to do when thread receives exception from work.
                    console.log("Got Exception");
                    this.handleWorkException(obj.id);
                } else {
                    this.doClientWork(obj);
                }
            });
        });
    },
    methods: {
        initThreads() {
            //Show normal viz
            let id=0;
            this.activeThreads.push({id: 0, step: 0});
        },
        directionVizThread() {
            let id = 0;
            let config = {channel: true, diversity: false, batch: 1, negative: this.showNegative};
            let style = {height: this.figHeight, width: this.figWidth};

            let socketParam = {id: id, layer: 'mixed4d_pre_relu', filter_idx: 0, style: style, config: config};
            this.socket.emit('startNewThread', socketParam);
            this.activeThreads.push({id: id, layer: 'l', filter_idx: '111', active: true, step: 0});
        },
        neuronsVizThread() {
            let id = 0;
            let config = {channel: true, diversity: false, batch: 1, negative: this.showNegative};
            let neuron1 = {layer: 'mixed4b_pre_relu', filterIndex: 111};
            let neuron2 = {layer: 'mixed4a_pre_relu', filterIndex: 476};
            let neurons = []
            neurons.push(neuron1)
            neurons.push(neuron2)

            let style = {height: this.figHeight, width: this.figWidth};
            // let socketParam = {id: id, layer: neuron.layer, filter_idx: neuron.filterIndex, style: style, config: config};
            let operation = 'add';
            let socketParam = {id: id, neurons: neurons, style: style, config: config, obj_op: operation};
            this.socket.emit('startNewThread', socketParam);
            this.activeThreads.push({id: id, layer: 'l', filter_idx: '111', active: true, step: 0});
        },
        setSocketParams(config, id, channel_index) {
            let style = {height: this.figHeight, width: this.figWidth};
            let socketParam = {id: id, filter_idx: channel_index, style: style, config: config};
            return socketParam
        },
        doClientWork(obj) {
            let imgData = obj.fig;
            let selectedThread = this.findActiveThread(obj.id);
            selectedThread.step += 1;
            selectedThread.figImg = imgData;
            if(selectedThread.step % 100 == 0) {
                this.socket.emit('save', obj.id);
                console.log("Image saved");
            }
            if(selectedThread.step < this.stepLimit) {
                setTimeout(() => {
                    if(!selectedThread.paused) {
                        this.socket.emit('perform', obj.id);  
                    }  
                }, 100);
            }
        },
        startNewThread(panelParam) {
            console.log(panelParam);
            let style = {height: this.figHeight, width: this.figWidth};
            panelParam.style = style;
            this.socket.emit('startNewThread', panelParam);
            //Save current filter index to check if changed in the middle
            this.currFilterIdx = panelParam.filter_idx;
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

<style scoped>
.mlp_div {
    background-color: aqua;
}
.mlp_div1 {
    background-color: aquamarine;
}
</style>