<template>
  <div class="container">
      <div class="header clearfix">
          <h3 class="text-muted">Object Detection Faster RCNN</h3>
      </div>
      <div class="jumbotron">
        <h3 class="jumbotronHeading">Upload an image</h3>
        <div class="canvasDiv">
          <picture-input ref="pictureInput" @change="onChange" width="200" height="200" margin="8" accept="image/jpeg,image/png" size="10"
              :removable="true" :customStrings="{
                upload: '<h1>Bummer!</h1>',
                drag: 'Drag a Image'
              }">
          </picture-input>
            <br />
            <p style="text-align:center;">
            <a class="btn btn-success myButton" @click="showMpld3" role="button">Predict</a>
            <a class="btn btn-primary" @click="clear" id="clearButton" role="button">Clear</a>
            </p>
        </div>
      </div>
      <div class="jumbotron">
        <p id="result"></p>
        <div class="slidecontainer">
          <!-- <p>Drag the slider to change the line width.</p>
          <input type="range" min="10" max="50" value="15" id="myRange" />
          <p>Value: <span id="sliderValue"></span></p> -->
          <img v-if="imageBytes!=''" v-bind:src="'data:image/jpeg;base64,'+imageBytes" />
          <div id="mlpcontainer" style="width:70%; height:400px;"></div>
        </div>
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
          htmlData: "Test"
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
      const path = `http://localhost:5000/query`;
      var qu = {"plot_type":"line"};
      axios.post(path, qu).then(res => {
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