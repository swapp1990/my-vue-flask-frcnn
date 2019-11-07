<template>
    <div>Saliency
        
    </div>
</template>

<script>
import axios from 'axios';

export default {
    name: "Saliency",
    data() {
        return {
            step: 0,
            threads: []
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