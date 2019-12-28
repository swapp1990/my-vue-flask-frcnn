<template>
    <div>
        <button :class="getConnectClass()" @click="connectSocket()"><i class="icon-magnet"></i></button>
        <div v-if="connected">
            <div class="p-2">
                <button @click="beginTraining()"><i class="icon-bolt"></i></button>
            </div>
            <div class="p-1">
                <div v-for="l in logs">{{l.log}}</div>
            </div>
            <div class="p-2">
                <img v-for="i in randomimgs" v-bind:src="'data:image/jpeg;base64,'+i" />
            </div>
            <div class="p-2">
                <span>Loss Graph</span>
                <div class="mlp_div" id="mlp_loss_graph"></div>
            </div>
            <div class="p-2">
                <div>
                    <button @click="loopEpochImages()"><i class="icon-forward"></i></button>
                    <button v-for="(e, i) in constImgsPerEpoch" @click="onEpochConstImageClick(i)">{{i}}</button>
                </div>
                <div>
                    <img v-for="i in constImgs" v-bind:src="'data:image/jpeg;base64,'+i" />
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import $ from 'jquery'
export default {
    name: "Pix2PixTrain",
    components: {
    },
    computed: {

    },
    data() {
        return {
            connected: false,
            socket: null,
            //Random image
            randomimgs: [],
            constImgs: [],
            constImgsPerEpoch: [],
            logs: []
        }
    },
    mounted(){
        this.connectSocket();
    },
    methods: {
        reset() {

        },
        getConnectClass() {
            let classes = [];
            if(this.connected) {
                classes.push("btn-primary");
            } else {
                classes.push("btn-danger");
            }
            return classes;
        },
        connectSocket() {
            this.socket = io.connect('http://127.0.0.1:5000');
            this.socket.on('connect',()=>{
                console.log("connected");
                this.onConnected();
            });
            this.socket.on('disconnect',()=>{
                console.log('disconnect');
                this.onDisconnected();
            });
            this.socket.on('connect_error', (error) => {
                console.log("Error");
                this.onDisconnected();
            });
            this.socket.on('error', (err) => {
                console.log("Error!", err);
            });
            this.socket.on('logs',(logs)=>{
                console.log(logs);
                this.handleLogs(logs);
            });
            this.socket.on('General',(content)=>{
                console.log('General ', content.action);
                this.handleGeneralMsg(content);
            });
        },
        onConnected() {
            this.socket.emit('init', this.training);
            this.connected = true;
            this.reset();
        },
        onDisconnected() {
            this.socket.close();
            this.connected = false;
        },
        handleGeneralMsg(content) {
            if(content.action) {
                if(content.action == "sendFigs") {
                    this.showFig(content.fig);
                } else if(content.action == "sendFigs2") {
                    this.showFigConstant(content.fig);
                } else if(content.action == "showGraph") {
                    this.showGraph(content.fig);
                }
            }
        },
        handleLogs(msg) {
            if(msg.logid) {
                if(msg.logid == "epoch") {
                    this.onEpochChange()
                }
                if(msg.type == "replace") {
                    let found = this.logs.find(l => {
                        return l.logid == msg.logid;
                    });
                    if(found) {
                        found.log = msg.log;
                    } else {
                        this.logs.push(msg);
                    }
                }
            } else {
                this.logs.push(msg);
            }
        },
        onEpochChange() {
            this.constImgsPerEpoch.push(this.constImgs);
        },

        // Pix2Pix
        beginTraining() {
            this.socket.emit('beginTraining');
        },
        showFig(fig) {
            // console.log(fig);
            this.randomimgs = [];
            fig.axes.forEach(a => {
                this.randomimgs.push(a.images[0].data);
            });
        },
        showFigConstant(fig) {
            this.constImgs = [];
            fig.axes.forEach(a => {
                this.constImgs.push(a.images[0].data);
            });
        },
        showGraph(imgData) {
            let mlpId = '#mlp_loss_graph';
            var graph1 = $(mlpId);
            graph1.html(imgData);
        },
        onEpochConstImageClick(idx) {
            this.constImgs = this.constImgsPerEpoch[idx];
        },
        loopEpochImages() {
            
        }
    }
}
</script>

<style lang="scss" scoped>
.mlp_div {
    background-color: aqua;
    width: 400px;
    height: 400px;
}
</style>
