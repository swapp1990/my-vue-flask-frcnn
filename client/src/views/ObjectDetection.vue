
<template>
  <div class="container-fluid">
    <div class="row">
      <!-- Container: PredictRPN -->
      <div class="container">
          <div class="row">
            <div class="col-md-12">
              <div class="header clearfix">
                <h3 class="text-muted">Object Detection Faster RCNN</h3>
              </div>
              <h3>Predict from Region Proposal Network (RPN)</h3>
              <p>Explain RPN</p>
            </div>
          </div>
        </div>
      </div>
      <!-- Demo: PredictRPN -->
      <div class="row">
        <div class="col-md-6">
          <pre> {{egcode1}} </pre>
        </div>
        <div class="col-md-6">
            ... your content here ...
        </div>
      </div>
      <!-- Container: RPN2ROI -->
      <div class="container">
        <div class="row">
          <div class="col-md-12">
            <h3>RPN to ROI</h3>
            <p>Explain RPN to ROI</p>
          </div>
        </div>
      </div>
      <!-- Demo: RPN2ROI -->
      <div class="row">
        <div class="col-md-6">
          <pre> {{egcode2}} </pre>
        </div>
        <div class="col-md-6">
          <div class="row">
            <div class="col-md-6">
              <a class="btn btn-success myButton" @click="showRpnToRoi" role="button">Predict</a>
            </div>
            <div class="col-md-6">
              <span v-for="item in allRatios">
                <input type="checkbox" :value="item.ratio" v-model="selectedRatios"> <span class="checkbox-label"> {{item.ratio}} </span>
              </span>
              <br>
              <span v-for="item in allSizes">
                <input type="checkbox" :value="item.size" v-model="selectedSizes"> <span class="checkbox-label"> {{item.size}} </span>
              </span>
            </div>
          </div>
          
          <div id="mlpcontainer" style="width:70%; height:400px;"></div>
          <div id="mlpcontainer2" style="width:70%; height:400px;"></div>
        </div>
      </div>
      <!-- Container: NonMaxSupression -->
      <div class="container">
        <div class="row">
          <div class="col-md-12">
            <h3>Non Max Suppression</h3>
            <p>Explain NonMaxSupression</p>
          </div>
        </div>
      </div>
      <!-- Demo: NonMaxSupression -->
      <div class="row">
        <!-- Code -->
        <div class="col-md-6">
          <pre> {{nms_code}} </pre>
        </div>
        <!-- Viz -->
        <div class="col-md-6">
          <div class="row">
            <div class="col-md-6">
              <a class="btn btn-success myButton" @click="showNonMax" role="button">Show All</a>
              <a class="btn btn-success myButton" @click="showSomeOverlap" role="button">Show Overlaps</a>
              <!-- <a class="btn btn-success myButton" @click="showSinglePyramidPool" role="button">Show Single</a> -->
            </div>
          </div>
          <div id="mlp_fig2_1" style="width:50%; height:300px;"></div>
          <div id="mlp_fig2_2" style="width:50%; height:300px;"></div>
          <div id="mlp_fig2_3" style="width:50%; height:300px;"></div>
        </div>
      </div>
      <!-- Container: PyramidPooling -->
      <div class="container">
        <div class="row">
          <div class="col-md-12">
            <h3>Spatial Pyramid Pooling</h3>
            <p>Explain Spatial Pyramid Pooling</p>
          </div>
        </div>
      </div>
      <!-- Demo: PyramidPooling -->
      <div class="row">
        <!-- Code -->
        <div class="col-md-6">
          <pre> {{egcode3}} </pre>
        </div>
        <!-- Viz -->
        <div class="col-md-6">
          <div class="row">
            <div class="col-md-6">
              <a class="btn btn-success myButton" @click="showFinalPyramidPooling" role="button">Show All</a>
              <a class="btn btn-success myButton" @click="showSinglePyramidPool" role="button">Show Single</a>
            </div>
          </div>
          <div id="mlp_fig3_2" style="width:70%; height:400px;"></div>
          <div id="mlp_fig3_1" style="width:70%; height:400px;"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import PictureInput from 'vue-picture-input'
import axios from 'axios';
import $ from 'jquery'
import * as code_txts from './code_text.js';

export default {
  name: "ObjDet",
  mounted() {
     
  },
  computed: {
      
  },
  data() {
      return {
          imageBytes: "",
          egcode1: `
          X = np.transpose(X, (0, 2, 3, 1))
          [cls_sigmoid, bbox_regr, base_layers] = model_rpn.predict(X)`,
          egcode2: code_txts.rpnTpRoi_code,
          egcode3: code_txts.SPP_code,
          nms_code: code_txts.NMS_code,
          htmlData: "Test",
          start: 0,
          end: 10,
          sizes: [128, 256, 512],
          allRatios: [
            { ratio: "1:1" },
            { ratio: "1:2" },
            { ratio: "2:1" }
          ],
          selectedRatios: [ '1:1' ] ,
          allSizes: [
            { size: "128" },
            { size: "256" },
            { size: "512" }
          ],
          selectedSizes: ["128"],
          nonMaxIdx: 0,
          poolIdx: 0,
          overlapIdx: 0
      }
  },
  components: {
      PictureInput
  },
  methods: {
    onChange(image) {
      console.log('New picture selected!')
      if (this.$refs.pictureInput.image) {
        console.log('Picture is loaded.')
        this.sendUploadToBackend(this.$refs.pictureInput.file.name, this.$refs.pictureInput.image)
      } else {
        console.log('FileReader API not supported: use the <form>, Luke!')
      }
    },
    sendUploadToBackend(name, data) {
      const path = 'http://localhost:5000/detect'

      axios.post(path, {'name': name, 'data': data})
      console.log('tried code in sendUploadToBackend')
    },
    predict() {
      const path = `http://localhost:5000/detect`
      axios.get(path)
      .then(response => {
        console.log(response)
        this.imageBytes = response.data;
      })
      .catch(error => {
        console.log(error)
      })
    },
    showRpnToRoi() {
      const path = `http://localhost:5000/getRois`;
      var params = {"start_range":this.start, "end_range":this.end, 
                    "ratios": this.selectedRatios,
                    "sizes": this.selectedSizes};
      axios.post(path, params).then(res => {
        var graph = $("#mlpcontainer");
        graph.html(res.data);
      })
      .catch(error => {
        console.log(error)
      })
    },
    showSinglePyramidPool() {
      const path = `http://localhost:5000/getSinglePyramidPool`;
      var params = {'poolIdx': this.poolIdx};
      axios.post(path, params).then(res => {
        var graph = $("#mlp_fig3_2");
        graph.html(res.data);
        this.poolIdx++;
        if(this.poolIdx == 10) this.poolIdx = 0;
      })
      .catch(error => {
        console.log(error)
      })
    },
    showFinalPyramidPooling() {
      const path = `http://localhost:5000/getPyramidPools`;
      var params = {};
      // var params = {"start_range":this.start, "end_range":this.end, 
      //               "ratios": this.selectedRatios,
      //               "sizes": this.selectedSizes};
      axios.post(path, params).then(res => {
        var graph = $("#mlp_fig3_1");
        graph.html(res.data);
      })
      .catch(error => {
        console.log(error)
      })
    },
    showNonMax() {
      const path = `http://localhost:5000/getNonmax`;
      
      var params = {"nonMaxIdx": this.nonMaxIdx}
      axios.post(path, params).then(res => {
        var mlp_figs = res.data
        var graph1 = $("#mlp_fig2_1");
        graph1.html(mlp_figs[0]);
        var graph2 = $("#mlp_fig2_2");
        graph2.html(mlp_figs[1]);
        var graph3 = $("#mlp_fig2_3");
        graph3.html(mlp_figs[2]);
        this.nonMaxIdx++;
        this.overlapIdx = 0;
      });

    },
    showSomeOverlap() {
      const path = `http://localhost:5000/getOverlap`;
      var params = {"overlapIdx": this.overlapIdx}
      axios.post(path, params).then(res => {
        var mlp_figs = res.data
        var graph1 = $("#mlp_fig2_1");
        graph1.html(mlp_figs[0]);
        var graph2 = $("#mlp_fig2_2");
        graph2.html(mlp_figs[1]);
        this.overlapIdx++;
      });
    },
    clear() {
      this.imageBytes = "";
    }
  }
}
</script>

<style scoped>
    body {
    padding-top: 20px;
    padding-bottom: 20px;
  }
  .header, .footer {
    padding-right: 15px;
    padding-left: 15px;
  }
  .header {
    padding-bottom: 20px;
    border-bottom: 1px solid #e5e5e5;
  }
  .header h3 {
    margin-top: 0;
    margin-bottom: 0;
    line-height: 40px;
  }
  .footer {
    padding-top: 19px;
    color: #777;
    border-top: 1px solid #e5e5e5;
  }
  @media (min-width: 768px) {
    .container {
      max-width: 730px;
    }
  }
  .container-narrow > hr {
    margin: 30px 0;
  }
  .jumbotron {
    text-align: center;
    border-bottom: 1px solid #e5e5e5;
    padding-top: 20px;
    padding-bottom: 20px;
  }
  .bodyDiv{
    text-align: center;
  }
  @media screen and (min-width: 768px) {
    .header,
    .footer {
      padding-right: 0;
      padding-left: 0;
  }
  .header {
    margin-bottom: 30px;
  }
  .jumbotron {
    border-bottom: 0;
  }
  }
  @media screen and (max-width: 500px) {
    .slidecontainer{
      display: none;
    }
  }
  .slidecontainer{
    float: left;
    width: 30%;
  }
  .jumbotronHeading{
    margin-bottom: 7vh;
  }
  .canvasDiv{
    display: flow-root;
    text-align: center;
    /* background-color: black; */
  }
</style>