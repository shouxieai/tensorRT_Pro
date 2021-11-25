// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from '@/App'
import router from '@/router/router'
import store from '@/vuex/vuex'
import Echo from '@/http/echo.js'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import "@/assets/bbicon/iconfont.css"
import VueCookies from 'vue-cookies'
import Tips from "@/tools/tips.js"
import Tools from "@/tools/tools.js"

Vue.use(ElementUI)
Vue.use(VueCookies)

//将需要的东西注入到全局
window["Echo"] = Echo
window["Tips"] = Tips
window["Tools"] = Tools

Vue.config.productionTip = false

//请求前的时候打开loading
let loadingLayer = null
Echo.setPreQueryCallback((query)=>{

  if(query._loadingSilence)
    return;

  if(loadingLayer != null) 
    return

  loadingLayer = ElementUI.Loading.service({
    fullscreen: true,
    lock: true,
    text: "加载中...",
    background: "rgba(0, 0, 0, 0.5)"
    //spinner: 'el-icon-loading',
    //target: ".data-content"
  })
})

//请求结束后，关闭loading
Echo.setEndQueryCallback((query, data)=>{

  if(loadingLayer != null){
    loadingLayer.close()
    loadingLayer = null
  }

  //如果不正确的结果返回，就提示错误
  if(data && data.status != "success"){
    if(!query._tipsSilence){
      Tips.error(data.message)
    }else{
      console.log("请求结束静默，不提示：" + data.msg)
    }
  }
})

new Vue({
  el: '#app',
  router,
  store,
  components: { App },
  template: '<App/>'
})