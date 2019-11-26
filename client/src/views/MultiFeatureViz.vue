<template>
    <div>
        <hr>
        <select v-model="selectedLayer">
            <option disabled value="">Select Layer: </option>
            <option v-for="l in inception_layers" v-bind:value="l.text">{{l.text}}</option>
        </select>
        <br>
        <button @click="addNewThread()"><i class="icon-plus"></i></button>
        <hr>

        <!-- <div class="stories">
            <div v-for="t in activeThreads" class="story">
                <viz-panel-simple :t="t"
                            v-on:create-thread="onCreateThread"></viz-panel-simple>
            </div>
        </div> -->
        <div v-for="layer in activeThreadsByLayer">
            <div class="header">
                <span>{{layer.type}}</span>
            </div>
            <div class="stories">
                <div v-for="t in layer.threads" class="story">
                    <viz-panel-simple :t="t"
                                v-on:create-thread="onCreateThread"></viz-panel-simple>
                </div>
            </div>
        </div>
        <hr>
    </div>
</template>

<script>
import {mapGetters} from 'vuex'
import $ from 'jquery'
import spinner from '@/components/Spinner'
import VizPanelSimple from '@/components/VizPanelSimple'

export default {
    name: "MultiFeat",
    components: {
        spinner: spinner,
        VizPanelSimple: VizPanelSimple
    },
    computed: {
        activeThreadsByLayer: function() {
            let ft = {};
            this.activeThreads.forEach(t => {
                if(!ft[t.layer]) {
                    ft[t.layer] = [];
                }
                ft[t.layer].push(t);
            });
            var ft_arr = [];
            for (var key in ft) {
                if (ft.hasOwnProperty(key)) {
                    ft_arr.push({"type" : key, "threads" : ft[key] })
                }
            }
            return ft_arr;
        }
    },
    data() {
        return {
            inception_layers: [
                { text: 'mixed4d'},
                { text: 'mixed4b_pre_relu'}
            ],
            currId: 0,
            selectedLayer: 'mixed4d',
            layers_map: {},
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
                console.log("thread started ", id);
                this.onThreadStarted(id);
            });
        //     this.socket.on('threadFinished', (id) => {
        //         console.log('threadFinished', id);
        //         this.removeActiveThread(id);
        //     });
            this.socket.on('workFinished', (obj) => {
                // console.log('workFinished', obj);
                this.doClientWork(obj);
            });
        //         if(obj.notFound) {
        //             console.log("Not Found");
        //         } else if(obj.exception) {
        //             //General Exception returned to Thread to handle what to do when thread receives exception from work.
        //             console.log("Got Exception");
        //             this.handleWorkException(obj.id);
        //         } else {
        //             this.doClientWork(obj);
        //         }
        //     });
        });
    },
    methods: {
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
            // if(selectedThread.step < this.stepLimit) {
            //     setTimeout(() => {
            //         if(!selectedThread.paused) {
            //             this.socket.emit('perform', obj.id);  
            //         }  
            //     }, 100);
            // }
        },
        startNewThread(panelParam) {
            console.log(panelParam);
            let style = {height: this.figHeight, width: this.figWidth};
            panelParam.style = style;
            this.socket.emit('startNewThread', panelParam);
            //Save current filter index to check if changed in the middle
            // this.currFilterIdx = panelParam.filter_idx;
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
.header {
	align-items:center;
	background:#FAFAFA;
	padding:$half-spacing-unit;
	border-bottom: 1px solid #eee;
    font-family: 'Dawning of a New Day', cursive;
    font-size: 24px;
    font-weight: bold;
}
.stories {
    border-bottom: 1px solid #f1f1f1;
    overflow-x: scroll;
    display: flex;
    padding: $half-spacing-unit;

    .story {
        flex: 0 0 auto;
        margin-right: $base-spacing-unit;
        text-align: center;
        &-userimage {
            border-radius: 50%;
            position: relative;
            border:2px solid red;
        }
    }
}
</style>