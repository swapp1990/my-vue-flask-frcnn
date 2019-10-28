
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
          egcode2: 
`def rpn_to_roi(rpn_layer, regr_layer, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9, debug=False):
    regr_layer = regr_layer / std_scaling

    (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            anchor_x = (anchor_size * anchor_ratio[0])/rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1])/rpn_stride
            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
            regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

            A[0, :, :, curr_layer] = X - anchor_x/2
            A[1, :, :, curr_layer] = Y - anchor_y/2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    #return top 300 bboxes
    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result`,
          egcode3: `
  for jk in range(R.shape[0]//num_rois + 1):
    pyramid_ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1), :], axis=0)
    if pyramid_ROIs.shape[1] == 0:
        break

    if jk == R.shape[0]//num_rois:
        continue

    [P_cls, P_regr] = model_classifier.predict([base_layers, pyramid_ROIs])
    for ii in range(P_cls.shape[1]):
      if np.max(P_cls[0, ii, :]) < G.bbox_threshold:
        continue
      max_cls_idx = np.argmax(P_cls[0, ii, :])
      (x, y, w, h) = pyramid_ROIs[0, ii, :]

      (tx, ty, tw, th) = P_regr[0, ii, 4*max_cls_idx:4*(max_cls_idx+1)]
      tx /= G.classifier_regr_std[0]
      ty /= G.classifier_regr_std[1]
      tw /= G.classifier_regr_std[2]
      th /= G.classifier_regr_std[3]
      x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)

      (x1, y1, x2, y2) = (x, y, w+x, h+y)
      `,
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
          poolIdx: 0
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
    showMpld3() {
      const path = `http://localhost:5000/getNonmax`;
      
      var params = {"nonMaxIdx": this.nonMaxIdx}
      axios.post(path, params).then(res => {
        var graph = $("#mlpcontainer");
        graph.html(res.data);
        this.nonMaxIdx++;
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