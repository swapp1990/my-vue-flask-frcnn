import Vue from 'vue';
import Router from 'vue-router';
import Books from './components/Books.vue';
import Ping from './components/Ping.vue';
import DrawMnist from './views/DrawMnist.vue';
import ObjDet from './views/ObjectDetection.vue';
import SocketTest from './views/SocketTest.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      component: SocketTest
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
