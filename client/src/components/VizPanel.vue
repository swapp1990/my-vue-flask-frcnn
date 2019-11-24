<template>
    <div class="row">
        <div class="col-sm-12">
            <hr>
            <select v-model="layer">
                <option disabled value="">Select Layer: </option>
                <option v-for="l in inception_layers" v-bind:value="l.text">{{l.text}}</option>
            </select>
            <br>
            <div class="input-group half">
                <div class="input-group-prepend">
                    <span class="input-group-text" id="">Filter Index</span>
                </div>
                <input type="number" class="form-control" v-model="filter_idx" @change="onFilterChange">
            </div>
            <br>
            Regularizer <input type="checkbox" v-model="config.use_regularizer">
            <div v-if="config.use_regularizer">
                <div class="input-group">
                    <div class="input-group-prepend">
                        <span class="input-group-text" id="">L1 (const & weight)</span>
                    </div>
                    <input type="number" step="0.1" class="form-control" v-model="config.regul.L1_const">
                    <input type="number" step="0.01" class="form-control" v-model="config.regul.L1_weight">
                </div>
                <div class="input-group half">
                    <div class="input-group-prepend">
                        <span class="input-group-text" id="">Total Variation (weight)</span>
                    </div>
                    <input type="number" step="0.25" min="-1" max="0" class="form-control" v-model="config.regul.TV_weight">
                </div>
                <div class="input-group half">
                    <div class="input-group-prepend">
                        <span class="input-group-text" id="">Blur Input (weight)</span>
                    </div>
                    <input type="number" step="0.01"  max="0" class="form-control" v-model="config.regul.Blur_weight">
                </div>
            </div>
            <hr>
            Thread {{t.id}}, {{layer}}:{{filter_idx}}
            <br>
            Step: {{t.step}}
            <br>
            <label for="checkbox">Use regularizer: {{ config.use_regularizer }}</label>
            <br>
            {{config.regul}}
            <br>
            <button v-if="!performing" class="btn" @click="perform(t.id)"><i class="icon-play"></i></button>
            <button v-if="performing" class="btn" @click="perform(t.id)"><i class="icon-pause"></i></button>
            <button class="btn" @click="stop(t.id)"><i class="icon-stop"></i></button>
            <button class="btn" @click="save(t.id)"><i class="icon-save"></i></button>
            <button class="btn" @click="load(t.id)"><i class="icon-archive"></i></button>
        </div>
        <div class="col-sm-12">
            <div class="mlp_div" :id="'mlp_fig_'+t.id" :style="figStyle"></div>
        </div>
    </div>
</template>
<script>
import $ from 'jquery'
export default {
    name: "VizPanel",
    props: ["t"],
    watch: {
      't.paused': { 
        handler(n, o) {
            console.log(n);
        },
        deep: true,
        immediate: true
      }
    },
    data() {
        return {
            started: false,
            performing: false,
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
            inception_layers: [
                { text: 'mixed4d'},
                { text: 'mixed4b_pre_relu'}
            ],
            layer: "mixed4b_pre_relu",
            filter_idx: 452,
        }
    },
    computed: {
        figStyle() {
            return 'width: ' + this.pwidth + 'px; height: ' + this.pheight + 'px;';
        }
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
        }
        
    },
    methods: {
        getPanelStyle() {
            let style = "width: " + this.pwidth + "px; height: " + this.pheight+ "px;";
            return style
        },
        perform(id) {
            if(!this.started) {
                let threadParam = {id: id, layer: this.layer, filter_idx: this.filter_idx, step:0, config: this.config}
                this.$emit("createThread", threadParam);
                this.started = true;
                this.performing = true;
            } else {
                this.performing = !this.performing;
                if(this.performing) {
                    let panelParam = {id: id, layer: this.layer, filter_idx: this.filter_idx, step:0, config: this.config}
                    let performParam = {id: id, f: this.filter_idx, panelParam: panelParam};
                    this.$emit("perform", performParam);
                } else {
                    this.$emit("pause", id);
                }
            }
        },
        stop(id) {
            this.performing = false;
            this.$emit("stop", id);
        },
        save(id) {
            this.performing = false;
            this.$emit("save", id);
        },
        load(id) {
            this.performing = false;
            let msg = {id: id, action: "load"};
            this.$emit("general", msg);
        },
        onFilterChange() {
            let msg = {id: this.t.id, action: "filterChange", filter_idx: this.filter_idx};
            this.$emit("general", msg);
        },
        showFig(fig) {
            // console.log(fig);
            var graph1 = $("mlp_fig");
            console.log(graph1);
            graph1.html(fig);
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
.mlp_div1 {
    background-color: aquamarine;
}
</style>