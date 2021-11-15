import Vue from 'vue'
import Router from 'vue-router'
import Echo from '@/http/echo'
import Error from "@/components/error"
import Home from "@/components/home"
import Tips from '@/tools/tips'
import { local } from 'd3'

Vue.use(Router)
const originalPush = Router.prototype.push

Router.prototype.push = function push(location) {
  return originalPush.call(this, location).catch(err => err)
}

const routes = [
  {
    path: "/",
    component: Home
  },
  {
    path: "*",
    component: Error
  }
]

const router = new Router({
  mode: 'history',
  routes: routes
})

router.beforeEach(function(to, from, next){
  return next()
})
export default router