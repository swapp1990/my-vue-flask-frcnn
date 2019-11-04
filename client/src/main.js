import 'bootstrap/dist/css/bootstrap.css';
import BootstrapVue from 'bootstrap-vue';
import Vue from 'vue';
import App from './App.vue';
import router from './router';
import VueSocketIO from 'vue-socket.io'
import { store } from './store'

Vue.use(BootstrapVue);
// Vue.use(new VueSocketIO({
//   debug: true,
//   connection: 'http://127.0.0.1:5000',
// }));

Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: h => h(App),
  created:function(){
    
  }
}).$mount('#app');
