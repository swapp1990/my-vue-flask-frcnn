<template>
    <div>
        Socket
    <button @click="clickButton">Click</button>
    <br>
    <span>Step: {{ this.step }}</span>
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
            step: 0
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
            this.socket.emit('firstclick');
        }
    }
}
</script>

<style scoped>

</style>