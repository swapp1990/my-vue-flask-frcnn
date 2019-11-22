<template>
    <div>
        <br>
        <input type="checkbox" id="checkbox" v-model="showNegative">
        <label for="checkbox">Show Negative: {{ showNegative }}</label>
        <button @click="initThreads()">Start Threads</button>
        <br>
        <div class="row">
            <div v-for="t in activeThreads" class="col-sm">
                Thread {{t.id}}, {{t.layer}}:{{t.filter_idx}}: <span>Step {{t.step}}</span>
                <div v-if="t.active">
                    <div class="spinner-border text-primary" role="status">
                        <span class="sr-only">Loading...</span>
                    </div>
                    <button @click="stopThread(t.id)">Stop</button>
                    <button @click="perform(t.id)">Perform</button>
                    <div class="mlp_div" :id="'mlp_fig_'+t.id" :style="figStyle"></div>
                </div>
                <span v-if="!t.active">Stopped</span>
            </div>
        </div>
        
        <!-- <button @click="clickButton">Click</button>
        <button @click="reset">Reset</button>
        <button @click="modify">Modify</button> -->
        <br>
        <!-- <span>Thread: {{this.threadC}} </span> -->
        <!-- <span>Step: {{ this.step }} </span> -->
        <!-- <span>prevStepImages: {{this.prevStepImages.length}}</span> -->
        <br>

        <!-- <select v-model="selected">
            <option disabled value="">Please select one</option>
            <option v-for="cls in imagenet_clss" v-bind:value="cls.value">{{cls.text}}</option>
        </select> -->
        <!-- <span>Selected: {{ selected }}</span> -->
        <br>
        <!-- <button v-for="img,i in prevStepImages" @click="showImg(img)">{{i}}</button> -->
        <!-- <button v-if="disableLive" @click="goLive()">Go Live</button>
        <button v-if="!disableLive" @click="showHighlights()">show Highlights</button> -->
    </div>
</template>

<script>
import {mapGetters} from 'vuex'
import $ from 'jquery'
import spinner from '@/components/Spinner'

export default {
    name: "Lucid2Test",
    components: {
        spinner: spinner
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
            // let filters = [24,67,86,123,45];
            //[426, 436, 43]
            let filters = [426, 436, 43];

            //Show normal viz
            let config = {channel: true, diversity: false, batch: 1, negative: this.showNegative};
            let id=0;
            let layer = "mixed4d";
            filters.forEach(f => {
                let params = {id: id, layer: layer, filterIndex: f, active: true, step: 0};
                this.activeThreads.push(params);
                let style = {height: this.figHeight, width: this.figWidth};
                let socketParam = {id: id, layer: layer, filter_idx: f, style: style, config: config};
                console.log(socketParam);
                this.socket.emit('startNewThread', socketParam);
                id++;
            });

            //Show diversity Viz
            // let config = {channel: true, diversity: true, batch: 4, negative: this.showNegative};
            // filters.forEach(f => {
            //     let params = {id: id, filterIndex: f, active: true, step: 0};
            //     this.activeThreads.push(params);
            //     let socketParams = this.setSocketParams(config, id, f);
            //     console.log(socketParams);
            //     this.socket.emit('startNewThread', socketParams);
            //     id++;
            // });

            //Show Directional Layer
            //this.directionVizThread();

            //Show Individual Neurons
            // this.neuronsVizThread();  
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

            let mlpId = '#mlp_fig_' + obj.id;
            var graph1 = $(mlpId);
            graph1.html(imgData);

            if(selectedThread.step < this.stepLimit) {
                setTimeout(() => {
                    this.socket.emit('perform', obj.id);    
                }, 200);
            }
        },
        //Thread
        startThread() {

        },
        perform(id) {
            this.socket.emit('perform', id);
        },
        stopThread(id) {
            this.socket.emit('stopThread', id);
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
            this.activeThreads[index].active = false;
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
</style>