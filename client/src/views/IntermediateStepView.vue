<template>
    <div>Thread
        <button @click="startThread()">start Thread</button>
        <button v-for="curr in this.threadsRunning" @click="incrementStep(curr)">Thread {{curr}}</button>
        <!-- <button @click="incrementStep()">Stop Thread</button> -->
        <!-- <span v-for="n in threads">Thread Finished - {{n}}\n</span> -->
    </div>
</template>

<script>
import axios from 'axios';

export default {
    name: "IntermediateStepView",
    data() {
        return {
            step: 0,
            threadsRunning: [],
            threadsStopped: []
        }
    },
    mounted(){
        this.socket = io.connect('http://127.0.0.1:5000');
        this.socket.on('connect',()=>{
            console.log("connceted");
        });
        this.socket.on('workFinished', (res) => {
            console.log('workFinished ', res.id);
            this.processThread(res);
        });
        this.socket.on('threadFinished', (name) => {
            console.log("thread finished ", name);
        });
        this.socket.on('threadStopped', (id) => {
            console.log('threadStopped ', id);
            this.threadsRunning = this.arrayRemove(this.threadsRunning, id);
        });
    },
    methods: {
        processThread(res) {
            if(res.step > 2) {
                this.stopThread(res.id);
            }
        },
        arrayRemove(arr, value) {
            return arr.filter(function(ele){
                return ele != value;
            });
        },
        startThread() {
            // const path = 'http://localhost:5000/startThread';
            // axios.post(path, {'name': this.step++});
            this.threadsRunning.push(this.step);
            this.socket.emit('startNewThread', this.step++);
        },
        incrementStep(curr) {
            this.socket.emit('incrementStep', curr);
        },
        stopThread(id) {
            this.socket.emit('stopThread', id);
            // this.socket.emit('stopThread');
            // 
            // console.log(this.threadsRunning);
        }
    }
}
</script>

<style>

</style>