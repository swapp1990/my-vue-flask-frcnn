<template>
    <div>
        <!-- <div class="row">
            <div class="mlp_div" style="width: 800px; height: 400px;"></div>
            <div class="mlp_div1" style="width: 800px; height: 400px;"></div>
        </div> -->
        <button @click="initThreads()">Start Threads</button>
        <div v-for="t in activeThreads" class="col-sm">
            <viz-panel :t="t"
                        v-on:createThread="onCreateThread"
                        v-on:perform="onPerform"
                        v-on:pause="onPause"
                        v-on:stop="onStop"></viz-panel>
        </div>
        <!-- <br>
        <input type="checkbox" id="checkbox" v-model="showNegative">
        <label for="checkbox">Show Negative: {{ showNegative }}</label>
        <button @click="initThreads()">Start Threads</button>
        <br>
        <div class="row">
            <div v-for="t in activeThreads" class="col-sm">
                <viz-panel :t="t"></viz-panel> -->
                <!-- Thread {{t.id}}, {{t.layer}}:{{t.filter_idx}}
                <br>
                Step: {{t.step}}
                <br>
                <div class="spinner-border text-primary" role="status">
                    <span class="sr-only">Loading...</span>
                </div>
                <button @click="stopThread(t.id)">Stop</button>
                <button @click="perform(t.id)">Perform</button>
                <div class="mlp_div" :id="'mlp_fig_'+t.id" :style="figStyle"></div> -->
            <!-- </div>
        </div> -->
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
        figStyle() {
            return 'width: ' + this.figWidth*2 + 'px; height: ' + this.figHeight + 'px;';
        }
    },
    data() {
        return {
            socket: null,
            step: 0,
            threadC: 0,
            selected: 20,
            stopped: false,
            prevStepImages: [],
            imagenet_clss: [
                { text: 'stingray', value: 6 },
                { text: 'magpie', value: 18 },
                { text: 'ouzel', value: 20 },
                { text: 'bullfrog', value:30},
                { text: 'turtle', value:37},
                { text: 'snake', value:55},
                { text: 'peacock', value:84}],
            disableLive: false,
            activeThreads: [],
            stepLimit: 500,
            figHeight: 400,
            figWidth: 400,
            showNegative: false
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
                // this.removeActiveThread(id);
                this.stopActiveThread(id);
            });
            this.socket.on('workFinished', (obj) => {
                // console.log('workFinished', obj);
                this.doClientWork(obj);
            });
        });
    },
    methods: {
        initThreads() {
            //Show normal viz
            let id=0;
            
            this.activeThreads.push({id: 0, step: 0});
            // let socketThreadParam = {id: id, layer: layer, filter_idx: filter_idx, step:0, style: style, config: config};
            // this.activeThreads.push(socketThreadParam);
            // this.socket.emit('startNewThread', socketThreadParam);
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

            if(selectedThread.step < this.stepLimit) {
                setTimeout(() => {
                    if(!selectedThread.paused) {
                        this.socket.emit('perform', obj.id);  
                    }  
                }, 1000);
            }
        },
        //Events
        onCreateThread(panelParam) {
            console.log(panelParam);
            let style = {height: this.figHeight, width: this.figWidth};
            panelParam.style = style;
            this.socket.emit('startNewThread', panelParam);
        },
        onPerform(id) {
            let selectedThread = this.findActiveThread(id);
            selectedThread.paused = false;
            this.socket.emit('perform', id);
        },
        onPause(id) {
            let selectedThread = this.findActiveThread(id);
            selectedThread.paused = true;
        },
        onStop(id) {
            this.socket.emit('stopThread', id);
        },
        stopThread(id) {
            this.socket.emit('stopThread', id);
            this.removeActiveThread(id);
        },
        findActiveThread(id) {
            var index = this.activeThreads.map(function(e) { return e.id; }).indexOf(id);
            if(index != -1)
                return this.activeThreads[id];
            else 
                return undefined
        },
        stopActiveThread(id) {
            var index = this.activeThreads.map(function(e) { return e.id; }).indexOf(id);
            if(this.activeThreads[index]) {
                this.activeThreads[index].active = false;
            }
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