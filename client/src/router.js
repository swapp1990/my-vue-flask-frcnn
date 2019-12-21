import Vue from 'vue';
import Router from 'vue-router';

import InceptionTrain from './views/InceptionTrainingViz.vue';
import TensorflowGan from './views/TensorflowGanViz.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      component: TensorflowGan
    }
  ],
});
