import Vue from 'vue';
import Router from 'vue-router';
import Books from './components/Books.vue';
import Ping from './components/Ping.vue';
import DrawMnist from './views/DrawMnist.vue';
import ObjDet from './views/ObjectDetection.vue';
import SocketTest from './views/SocketTest.vue';
import LucidTest from './views/LucidTest.vue';
import Lucid2Test from './views/Lucid2Images.vue';
import ThreadTest from './views/ThreadTest.vue';
import Saliency from './views/Saliency.vue';
import IntermediateStepView from './views/IntermediateStepView.vue';
import FeatureViz from './views/FeatureViz.vue';
import SingleFeat from './views/SingleFeatureViz.vue';
import MultiFeat from './views/MultiFeatureViz.vue'
import FilteredTabs from './views/workingUiSamples/FilteredImageTabs.vue'
import InstaView from './views/workingUiSamples/InstaView.vue';
import ComparifyView from './views/workingUiSamples/ComparifyView.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      component: MultiFeat
    },
    {
      path: '/obj',
      name: 'ObjDet',
      component: ObjDet,
    },
    {
      path: '/draw',
      name: 'DrawMnist',
      component: DrawMnist,
    },
    {
      path: '/ping',
      name: 'Ping',
      component: Ping,
    },
  ],
});
