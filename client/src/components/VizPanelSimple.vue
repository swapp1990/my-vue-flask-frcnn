<template>
    <div class="row">
        <div class="col-sm-12">
            <div class="mlp_div" :id="'mlp_fig_'+t.id" :style="figStyle"></div>
            {{t.id}}:{{t.filter_idx}} | Step: {{t.step}}
        </div>
    </div>
</template>
<script>
import $ from 'jquery'
export default {
    name: "VizPanelSimple",
    mounted() {
        // console.log(this.t);
        this.create();
    },
    props: ["t"],
    computed: {

    },
    watch: {
        t: {
            handler (n, o) {
                // console.log(n);
                if(n.figImg) {
                    this.displayImg(n);
                }
            },
            deep: true,
            immediate: true
        },
        't.filter_idx': { 
            handler(n, o) {
                // console.log(n);
            },
            deep: true,
            immediate: true
        }
    },
    data() {
        return {
            pwidth: 300,
            pheight: 300,
            config: {
                channel: true, 
                diversity: false, 
                batch: 1, 
                negative: false,
                use_regularizer: true,
                regul: {
                    L1_const: .5,
                    L1_weight: -0.05,
                    TV_weight: -0.25,
                    Blur_weight: 0,
                }
            },
        }
    },
    computed: {
        figStyle() {
            return 'width: ' + this.pwidth + 'px; height: ' + this.pheight + 'px;';
        }
    },
    methods: {
        create() {
            let threadParam = {id: this.t.id, layer: this.t.layer, step:this.t.step, 
                                filter_idx: this.t.filter_idx, config: this.config}
            this.$emit("create-thread", threadParam);
        },
        displayImg(content) {
            let mlpId = '#mlp_fig_' + content.id;
            var graph = $(mlpId);
            graph.html(content.figImg);
        }
    }
}
</script>
<style scoped>
.half {
  width: 50%;
}
.mlp_div {
    background-color: aqua;
}
</style>