
<template>
  <div class="container">
      <div class="header clearfix">
          <h3 class="text-muted">Object Detection Faster RCNN</h3>
      </div>
      <div class="jumbotron">
        <div class="row">
          <div class="col-sm">
            <div class="input-group input-group-sm p-3">
              <input v-model="start" placeholder="0">
              <input v-model="end" placeholder="10">
            </div>
            <a class="btn btn-success myButton" @click="showMpld3" role="button">Predict</a>
            <a class="btn btn-primary" @click="clear" id="clearButton" role="button">Clear</a>
          </div>
          <div class="col-sm">
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
        <!-- <h3 class="jumbotronHeading">Upload an image</h3>
        <div class="canvasDiv">
          <picture-input ref="pictureInput" @change="onChange" width="200" height="200" margin="8" accept="image/jpeg,image/png" size="10"
              :removable="true" :customStrings="{
                upload: '<h1>Bummer!</h1>',
                drag: 'Drag a Image'
              }">
          </picture-input>
            <br />
            <p style="text-align:center;">
            
            </p>
        </div> -->
      </div>
  </div>
</template>

<script>
import PictureInput from 'vue-picture-input'
import axios from 'axios';
import $ from 'jquery'
export default {
  name: "ObjDet",
  mounted() {
     
  },
  computed: {
      
  },
  data() {
      return {
          imageBytes: "",
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
          selectedSizes: ["128"]
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
    showMpld3() {
      const path = `http://localhost:5000/getRois`;
      var params = {"start_range":this.start, "end_range":this.end, 
                    "ratios": this.selectedRatios,
                    "sizes": this.selectedSizes};
      axios.post(path, params).then(res => {
        var graph = $("#mlpcontainer");
        graph.html(res.data);
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