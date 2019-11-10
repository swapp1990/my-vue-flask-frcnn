<template>
    <div>
        <!-- <h1 class="ml1">
            <span class="text-wrapper">
                <span class="line line1"></span>
                <span class="letters">SOCKET</span>
                <span class="line line2"></span>
            </span>
        </h1> -->
        <br>
        <button @click="initThreads()">Start Threads</button>
        <!-- <button @click="startLucid()">Start</button>
        <button @click="nextLucid()">Next</button>
        <button @click="stopLucid()">Stop</button>
        <input v-model="filter_idx"> -->
        <br>
        <!-- <button @click="startThread()">Start Thread</button>
        <input v-model="filter_idx"> -->
        <div class="row">
            <div v-for="t in activeThreads" class="col-sm">
                Thread {{t.id}}, Filter Index {{t.filterIndex}}: <span>Step {{t.step}}</span>
                <div v-if="t.active">
                    <div class="spinner-border text-primary" role="status">
                        <span class="sr-only">Loading...</span>
                    </div>
                    <button @click="stopThread(t.id)">Stop</button>
                    <button @click="perform(t.id)">Perform</button>
                    <div class="mlp_div" :id="'mlp_fig_'+t.id" style="width:640px; height:480px;"></div>
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
    name: "LucidTest",
    components: {
        spinner: spinner
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
            stepLimit: 50
        }
    },
    mounted(){
        // this.initText();
        this.step = 0;
        // this.$store.dispatch("SET_CHAT");
        // this.sockets.subscribe('customEmit', (data) => {
        //     console.log(data);
        // });
        this.socket = io.connect('http://127.0.0.1:5000');
        this.socket.on('connect',()=>{
            console.log("connected");
            this.socket.on('gotfig', (data) => {
                if(!this.stopped) {
                    this.socket.emit('nextImg', this.step++);
                    console.log("gotfig ", this.step);
                    var graph1 = $("#mlp_fig");
                    graph1.html(data);
                }
            });
            this.socket.on('test', (id) => {
                console.log("test ", id);
            });
            this.socket.on('threadFinished', (id) => {
                console.log('threadFinished', id);
                // this.removeActiveThread(id);
                this.stopActiveThread(id);
            });
            this.socket.on('workFinished', (obj) => {
                console.log('workFinished', obj);
                this.doClientWork(obj);
            });
        });
    },
    methods: {
        initThreads() {
            let filters = [24,67,86,123,45];
            let id = 0;
            filters.forEach(f => {
                let params = {id: id, filterIndex: f, active: true, step: 0};
                this.activeThreads.push(params);
                let socketParam = {id: params.id, filterIndex: params.filterIndex};
                console.log(socketParam);
                this.socket.emit('startNewThread', socketParam);
                id++;
            });
        },
        initText() {
            var textWrapper = document.querySelector('.ml1 .letters');
            textWrapper.innerHTML = textWrapper.textContent.replace(/\S/g, "<span class='letter'>$&</span>");

            anime.timeline({loop: true})
            .add({
                targets: '.ml1 .letter',
                scale: [0.3,1],
                opacity: [0,1],
                translateZ: 0,
                easing: "easeOutExpo",
                duration: 600,
                delay: (el, i) => 70 * (i+1)
            }).add({
                targets: '.ml1 .line',
                scaleX: [0,1],
                opacity: [0.5,1],
                easing: "easeOutExpo",
                duration: 700,
                offset: '-=875',
                delay: (el, i, l) => 80 * (l - i)
            }).add({
                targets: '.ml1',
                opacity: 0,
                duration: 1000,
                easing: "easeOutExpo",
                delay: 1000
            });
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
                }, 100);
            }
        },
        reset() {
            this.socket.emit('reset');
            // this.step = 500;
        },
        modify() {
            let modifyParams = {"l2_norm": true, "cls_idx": this.selected}
            this.socket.emit('modify', modifyParams);
        },
        startLucid() {
            this.step = 0;
            this.stopped = false;
            this.socket.emit('startLucid', this.filter_idx);
        },
        nextLucid() {
            this.socket.emit('nextImg', this.step);
        },
        stopLucid() {
            this.stopped = true;
        },
        startThread() {
            //if(!this.filter_idx) this.filter_idx = 20;
            
            this.threadC += 1;
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
        clickButton: function () {
            //this.socket.emit('first-connect1','clicked user has connected');
            //console.log("emit");
            console.log(this.selected);
            this.socket.emit('firstclick', this.selected);
            this.step = 0;
        },
        showImg(imgData) {
            this.disableLive = true;
            var graph1 = $("#mlp_fig");
            graph1.html(imgData);
        },
        goLive() {
            this.disableLive = false;
            clearInterval(this.intervalid1);
        },
        showHighlights() {
            this.disableLive = true;
            let i = 0;
            this.intervalid1 = setInterval(() => {
                i++;
                if(i == this.prevStepImages.length) i = 0;
                this.showImg(this.prevStepImages[i]);
            }, 200);
        }
    }
}
</script>

<style scoped>
.mlp_div {
    background-color: aqua;
}
.ml1 {
  font-weight: 900;
  font-size: 3.5em;
}

.ml1 .letter {
  display: inline-block;
  line-height: 1em;
}

.ml1 .text-wrapper {
  position: relative;
  display: inline-block;
  padding-top: 0.1em;
  padding-right: 0.05em;
  padding-bottom: 0.15em;
}

.ml1 .line {
  opacity: 0;
  position: absolute;
  left: 0;
  height: 3px;
  width: 100%;
  background-color: #fff;
  transform-origin: 0 0;
}

.ml1 .line1 { top: 0; }
.ml1 .line2 { bottom: 0; }
</style>