<template>
    <div>
        Saliency
        <upload-image v-on:uploaded="onUpload()"></upload-image>
        <button @click="showGradCam()">Show Grad Cam</button>
        <button @click="showSaliency()">show Saliency</button>
        <div id="mlp_fig" style="width:70%; height:400px;"></div>
    </div>
</template>

<script>
import axios from 'axios';
import uploadImage from '../components/Upload';
import $ from 'jquery'

export default {
    name: "Saliency",
    components: {
        uploadImage: uploadImage
    },
    data() {
        return {
            step: 0,
            threads: [],
            gotImage: false
        }
    },
    mounted(){
        // this.socket = io.connect('http://127.0.0.1:5000');
        // this.socket.on('connect',()=>{
        //     console.log("connceted");
        // });
        // this.socket.on('bg_emit', () => {
        //     console.log("bg emit");
        // });
        // this.socket.on('threadFinished', (name) => {
        //     console.log("thread finished ", name);
        //     this.threads.push(name);
        // });
    },
    methods: {
        onUpload() {
            gotImage = true;
        },
        showGradCam() {
            const path = `http://localhost:5000/getGradCam`;
            var params = {}
            axios.post(path, params).then(res => {
                var mlp_figs = res.data;
                var graph1 = $("#mlp_fig");
                graph1.html(mlp_figs[0]);
            });
        },
        showSaliency() {
            const path = `http://localhost:5000/getSaliency`;
            var params = {}
            axios.post(path, params).then(res => {
                var mlp_figs = res.data;
                var graph1 = $("#mlp_fig");
                graph1.html(mlp_figs[0]);
            });
        },
        startThread() {
            // const path = 'http://localhost:5000/startThread';
            // axios.post(path, {'name': this.step++});

            this.socket.emit('startNewThread', this.step++);
        }
    }
}
</script>

<style>

</style>