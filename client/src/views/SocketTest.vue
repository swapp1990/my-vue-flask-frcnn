<template>
    <div>
        Socket
    <button @click="clickButton">Click</button>
    <br>
    <span>Step: {{ this.step }}</span>
    <br>
    <select v-model="selected">
        <option disabled value="">Please select one</option>
        <option v-for="cls in imagenet_clss" v-bind:value="cls.value">{{cls.text}}</option>
    </select>
    <span>Selected: {{ selected }}</span>
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
            imagenet_clss: [
                { text: 'stingray', value: 6 },
                { text: 'magpie', value: 18 },
                { text: 'ouzel', value: 20 },
                { text: 'bullfrog', value:30},
                { text: 'turtle', value:37},
                { text: 'snake', value:55}]
            }
    },
    mounted(){
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
                    var graph1 = $("#mlp_fig");
                    graph1.html(data);
                }, 50);
                
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
        clickButton: function () {
            //this.socket.emit('first-connect1','clicked user has connected');
            //console.log("emit");
            console.log(this.selected);
            this.socket.emit('firstclick', this.selected);
            this.step = 0;
        }
    }
}
</script>

<style scoped>

</style>