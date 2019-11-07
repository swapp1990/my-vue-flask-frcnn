<template>
    <div>
        Socket
        <button @click="clickButton">Click</button>
        <button @click="reset">Reset</button>
        <button @click="modify">Modify</button>
        <br>
        <span>Step: {{ this.step }} </span>
        <span>prevStepImages: {{this.prevStepImages.length}}</span>
        <br>
        <select v-model="selected">
            <option disabled value="">Please select one</option>
            <option v-for="cls in imagenet_clss" v-bind:value="cls.value">{{cls.text}}</option>
        </select>
        <span>Selected: {{ selected }}</span>
        <br>
        <!-- <button v-for="img,i in prevStepImages" @click="showImg(img)">{{i}}</button> -->
        <button v-if="disableLive" @click="goLive()">Go Live</button>
        <button v-if="!disableLive" @click="showHighlights()">show Highlights</button>
        <div id="mlp_fig" style="width:70%; height:400px;"></div>
    </div>
</template>

<script>
import {mapGetters} from 'vuex'
import $ from 'jquery'

export default {
    name: "SocketTest",
    data() {
        return {
            socket: null,
            step: 0,
            selected: 20,
            prevStepImages: [],
            imagenet_clss: [
                { text: 'stingray', value: 6 },
                { text: 'magpie', value: 18 },
                { text: 'ouzel', value: 20 },
                { text: 'bullfrog', value:30},
                { text: 'turtle', value:37},
                { text: 'snake', value:55},
                { text: 'peacock', value:84}],
            disableLive: false
        }
    },
    mounted(){
        this.step = 0;
        // this.$store.dispatch("SET_CHAT");
        // this.sockets.subscribe('customEmit', (data) => {
        //     console.log(data);
        // });
        this.socket = io.connect('http://127.0.0.1:5000');
        this.socket.on('connect',()=>{
            console.log("connceted");
            //this.socket.emit('first-connect1','Socket test user has connected');
            this.socket.emit('first-connect1');
            this.socket.on('optinit', (data) => {
                console.log("Opt Init");
                this.socket.emit('performminimize', this.step++);
            })
            this.socket.on('gotfig', (data) => {
                setTimeout(() => {
                    // console.log('Got Fig');
                    this.socket.emit('performminimize', this.step++);
                    if(!this.disableLive) {
                        var graph1 = $("#mlp_fig");
                        graph1.html(data);
                    }
                    // if(this.step % 5 == 0) {
                    //     this.prevStepImages.push(data);
                    // }
                }, 150);
                
            });
        });
    },
    // computed:{
    // ...mapGetters(['CHATS','HANDLE'])
    // },
    // sockets : {
    //     connect: function(){
    //         console.log('socket connected');
    //         this.$socket.emit('ping');
    //     },
    //     customEmit: function (data) {
    //         console.log('this method was fired by the socket server. eg: io.emit("customEmit", data)')
    //     }
    // },
    methods: {
        reset() {
            this.socket.emit('reset');
            // this.step = 500;
        },
        modify() {
            let modifyParams = {"l2_norm": true, "cls_idx": this.selected}
            this.socket.emit('modify', modifyParams);
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

</style>