<template>
<div>
    <div id="top-panel">
        <div id="top-panel-container" class="panel-container">
            <div id="page-title" class="top-panel-item"><strong>GAN</strong> Lab</div>
            <div id="timeline-controls" class="top-panel-item">
                <button 
                    class="mdl-button mdl-js-button mdl-button--icon ui-resetButton"
                    id="reset-button" title="Reset the model">
                    <i class="material-icons" id="button-top-reset">replay</i>
                </button>
                <button class="mdl-button mdl-js-button mdl-button--fab 
                    mdl-button--colored ui-playButton"
                    id="play-pause-button" title="Run/Pause training">
                    <i class="material-icons" id="button-top-play">play_arrow</i>
                    <i class="material-icons" id="button-top-pause">pause</i>
                </button>
                
            </div>
            <div id="iteration" class="top-panel-item">
                <div class="top-column-title">Epoch</div>
                <div id="iteration-count">0</div>
            </div>
        </div>
    </div>
    <div id="main-panel">
        <div id="main-panel-container" class="panel-container">
            <div id="model-visualization-container" class="panel-item">
                <div class="panel-title" style="padding-left: 5px;">
                    Model Overview Graph
                    <button class="mdl-button mdl-js-button mdl-button--icon"
                        id="edit-model-button" title="Show/hide hyperparameters">
                        <i class="material-icons" id="button-graph-edit">mode_edit</i>
                    </button>
                </div>
                <div id="model-vis-content-container">
                    <svg id="model-vis-svg" class="">
                        <path d="M66,260 L102,260"
                            id="arrow-g-forward-i" class="d-update-flow g-update-flow" />
                        <path d="M248,260 L284,260"
                            id="arrow-g-forward-o" class="d-update-flow g-update-flow" />
                        <path d="M336,166 L372,166"
                            id="arrow-t-d-forward-i" class="d-update-flow" />
                        <path d="M518,166 L554,166"
                            id="arrow-t-d-forward-o" class="d-update-flow" />
                        <path d="M336,260 L372,260"
                            id="arrow-g-d-forward-i" class="d-update-flow g-update-flow" />
                        <path d="M518,260 L554,260"
                            id="arrow-g-d-forward-o" class="d-update-flow g-update-flow" />

                        <path d="M605,166 C625,166 615,213 640,213"
                            id="arrow-t-prediction-d-loss" class="d-update-flow" />
                        <path d="M605,260 C625,260 615,213 640,213"
                            id="arrow-g-prediction-d-loss" class="d-update-flow" />
                        <path d="M640,213 L650,213 a10,10 0 0 0 10,-10
                            L660,40 a10,10 0 0 0 -10,-10
                            L445,30 a10,10 0 0 0 -10,10 L435,141"
                            id="arrow-d-loss-d" class="d-update-flow" />

                        <path d="M605,260 L650,260 a10,10 0 0 1 10,10
                            L660,415 a10,10 0 0 1 -10,10 L336,425"
                            id="arrow-g-loss-g-1" class="g-update-flow" />
                        <path d="M289,425 L185,425 a10,10 0 0 1 -10,-10 L175,303"
                            id="arrow-g-loss-g-2" class="g-update-flow" />
                    </svg>

                    <div id="model-vis-div">
                        <div id="component-d-loss" class="model-component" style="top: 160px; left: 667px">
                            <div id="d-loss-bar" class="loss-bar" title=""></div>
                            <div class="component-loss-label">Discriminator<br />loss</div>
                            <div class="component-tooltip">
                            <span id="d-loss-value" title="loss"></span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
</template>
<script>
import {mapGetters} from 'vuex'
import $ from 'jquery'

export default {
    name: "TensorflowGan",
}
</script>
<style scoped>
    #top-panel {
      background-color: #eee;
      border-bottom: 2px solid #ddd;
      font-size: 30px;
      padding: 10px 0 9px;
    }
    #top-panel-container {
      align-items: center;
    }
    .top-panel-item {
      margin-left: 30px;
      margin-right: 40px;
    }
    #main-panel {
      background-color: #f7f7f7;
      border-bottom: 2px solid #eee;
    }
    #model-visualization-container {
      border: 1px solid rgba(0, 0, 0, 0.1);
      border-width: 0 1px 0 0;
      min-height: 570px;
      min-width: 740px;
    }
    .panel-container {
        display: flex;
        margin-left: auto;
        margin-right: auto;
        justify-content: space-between;
        width: 1442px;
        /* background-color: aqua; */
    }
    .panel-item {      
      padding: 15px 20px;
    }
    .panel-title {
      align-items: center;
      color: #555;
      display: flex;
      margin-bottom: 5px;
      font-size: 15px;
      height: 32px;
      text-transform: uppercase;
    }

    .panel-title .mdl-button--icon {
      margin-left: 6px;
    }

    #timeline-controls {
      align-items: center;
      display: flex;
    }
    #timeline-controls button {
      margin-right: 11px;
    }
    .mdl-button--fab.mdl-button--colored,
    .mdl-button--fab.mdl-button--colored:hover,
    .mdl-button--fab.mdl-button--colored:active,
    .mdl-button--fab.mdl-button--colored:focus,
    .mdl-button--fab.mdl-button--colored:focus:not(:active) {
      background: #183D4E;
    }
    #play-pause-button .material-icons {
      color: white;
      font-size: 36px;
      transform: translate(-18px,-12px);
    }
    #play-pause-button .material-icons:nth-of-type(2) {
      display: none;
    }
    #play-pause-button.playing .material-icons:nth-of-type(1) {
      display: none;
    }
    #play-pause-button.playing .material-icons:nth-of-type(2) {
      display: inherit;
    }
    #iteration {
      flex-grow: 1;
      font-size: 14px;
      margin-left: 20px;
    }
    #iteration .top-column-title {
      color: #222;
      font-size: 13px;
    }
    #iteration-count {
      color: #666;
      font-size: 28px;
      margin: 5px 0 13px;
    }

    /* SVG COntainer */
    #model-vis-content-container {
      position: relative;
    }
    #model-vis-svg,
    #model-vis-div {
      height: 570px;
      position: absolute;
      width: 740px;
    }

    #model-vis-svg {
      pointer-events: none;
    }

    #model-vis-svg path {
      fill: none;
      stroke: rgb(175, 175, 175);
      stroke-dasharray: 10,2;
      stroke-width: 2;
    }

    #model-vis-svg path.d-update-flow.d-activated {
      stroke: rgb(105, 158, 255);
      stroke-width: 4;
    }

    #model-vis-svg path.g-update-flow.g-activated {
      stroke: rgb(186, 99, 207);
      stroke-width: 4;
    }

    #model-vis-svg.playing path.d-activated,
    #model-vis-svg.playing path.g-activated {
      animation: dash 3600s linear forwards;
    }
    /* model component */
    .model-component {
      padding-top: 5px;
      position: absolute;
      text-align: center;
      width: 110px;
    }
    .model-component svg {
      background-color: #fff;
      border: 1px solid #eee;
      border-radius: 3px;
      height: 50px;
      width: 50px;
    }
    .component-loss-label {
      color: rgb(121, 121, 121);
      font-size: 12px;
      line-height: 12px;
      margin-top: 3px;
      text-align: left;
    }
    .loss-bar {
      height: 12px;
      width: 0;
    }
    #d-loss-bar {
      background-color: rgb(105, 158, 255);
    }
</style>