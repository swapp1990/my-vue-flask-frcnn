<template>
    <div class="row">
        <div class="col-sm-12">
            <span>{{imgCacheidx}},{{gifSpeed}},{{imgArr.length}}</span>
        </div>
        <hr>
        <div class="col-sm-12">
            <div class="row">
                <div class="col-sm-4">
                    <label for="customRange1">Speed</label>
                    <input type="range" class="custom-range" id="customRange1" min="100" max="1000" step="50"
                            v-on:change="changeSpeed()" v-model="gifSpeed">
                </div>
            </div>
        </div>
        <div class="col-sm-12">
            <img v-bind:src="'data:image/jpeg;base64,'+this.imgCurr" />
        </div>
    </div>
</template>
<script>
export default {
    name: "imagesGif",
    props: ['imgArr'],
    data() {
        return {
            imgCurr: null,
            imgCacheidx: 0,
            gifSpeed: 500,
            myInterval: null,
            imgRemoveIndex: 1,
            removedImgs: 0,
            maxArrLen: 20
        }
    },
    watch: {
        'imgArr': function(val, old) {
            if(val.length == 1) {
                this.startGif();
            }
            this.processArr();
        },
        immediate: true
    },
    methods: {
        processArr() {
            if(this.imgArr.length > this.maxArrLen) {
                this.$emit('splice', this.imgRemoveIndex);
                this.removedImgs++;
                if(this.removedImgs % 10 == 0) {
                    this.imgRemoveIndex++;
                }
            }
        },
        startGif() {
            this.imgCacheidx = 0;
            if(this.myInterval != null) {
                clearInterval(this.myInterval);
            }
            this.myInterval = setInterval(() => {
                this.imgCacheidx++;
                if(this.imgArr[this.imgCacheidx]) {
                    this.imgCurr = this.imgArr[this.imgCacheidx];
                } else {
                    this.imgCacheidx = 0;
                }
            }, this.gifSpeed);
        },
        changeSpeed() {
            // console.log(this.gifSpeed);
            this.startGif();
        }
        
    }
}
</script>
<style scoped>
    img {
        max-width: 100%;
        max-height: 100%;
    }
</style>