<template>
    <div class="row">
        <div class="col-sm-12">
            <input type="number" v-model="filter_idx" @change="onFilterChange">
            <br>
            Thread {{t.id}}, {{layer}}:{{filter_idx}}
            <br>
            Step: {{t.step}}
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
    data() {
        return {
            started: false,
            performing: false,
            pwidth: 300,
            pheight: 300,
            config: {channel: true, diversity: false, batch: 1, negative: false},
            layer: "mixed4d",
            filter_idx: 423,
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
.mlp_div {
    background-color: aqua;
}
.mlp_div1 {
    background-color: aquamarine;
}
</style>