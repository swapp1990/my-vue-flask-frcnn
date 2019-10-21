<template>
  <div class="container">
      <div class="header clearfix">
        <h3 class="text-muted">MNIST Handwritten CNN</h3>
      </div>
      <div class="jumbotron">
        <h3 class="jumbotronHeading">Draw the digits inside the box</h3>
        <div class="slidecontainer">
          <p>Drag the slider to change the line width.</p>
          <input type="range" min="10" max="50" value="15" id="myRange" />
          <p>Value: <span id="sliderValue"></span></p>
        </div>

        <div class="canvasDiv">
            <canvas id="canvas" width="280" height="280" style="padding-bottom: 20px"
              v-on:mousedown="handleMouseDown" 
              v-on:mouseup="handleMouseUp" 
              v-on:mousemove="handleMouseMove" >
            </canvas>
            <br />
            <p style="text-align:center;">
            <a class="btn btn-success myButton" @click="predict" role="button">Predict</a>
            <a class="btn btn-primary" @click="clear" id="clearButton" role="button">Clear</a>
            </p>
        </div>
      </div>
      <div class="jumbotron">
        <p id="result">{{prediction}}</p>
      </div>
  </div>
</template>

<script>
import axios from 'axios';
export default {
  name: "DrawMnist",
  mounted() {
      var c = document.getElementById("canvas");
      this.context = c.getContext("2d");
      this.context.fillStyle = "black";
      this.context.fillRect(0, 0, c.width, c.height);
  },
  computed: {
      currentMouse() {
          var c = document.getElementById("canvas");
          var rect = c.getBoundingClientRect();
          
          return {
              x: this.mouse.current.x - rect.left,
              y: this.mouse.current.y - rect.top
          }
      }
  },
  data() {
      return {
          context: null,
          mouse: {
              current: {
                  x: 0,
                  y: 0
              },
              previous: {
                  x: 0,
                  y: 0
              },
              down: false
          },
          prediction: "Get your prediction here!!!"
      }
  },
  methods: {
    clear() {
      var c = document.getElementById("canvas");
      this.context = c.getContext("2d");
      this.context.clearRect(0, 0, c.width, c.height);
      this.context.fillStyle = "black";
      this.context.fillRect(0, 0, c.width, c.height);
    },
    predict() {
      const canvas = document.getElementById('canvas');
      var img = canvas.toDataURL();
      let config = {
        header : {
          'Content-Type' : 'multipart/form-data',
          'Acces-Control-Allow-Origin': '*'
        }
      }
      axios
        .post('http://localhost:5000/predict', img, config)
        .then(response => {
          console.log(response.data);
          this.prediction = response.data.pred + " with probability of " + response.data.prob;
        });
      
    },
    draw(e) {
        if (this.mouse.down ) {
          this.context.lineTo(this.currentMouse.x, this.currentMouse.y);
          this.context.strokeStyle ="#F63E02";
          this.context.lineWidth = 2;
          this.context.stroke()
        }
    },
    handleMouseDown(e) {
      this.mouse.down = true;
      this.mouse.current = {
        x: e.pageX,
        y: e.pageY
      }

      this.context.beginPath();
      this.context.moveTo(this.currentMouse.x, this.currentMouse.y)
    },
    handleMouseUp() {
        this.mouse.down = false;
    },
    handleMouseMove(e) {
      this.mouse.current = {
        x: event.pageX,
        y: event.pageY
      }
      this.draw(e);
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