import axios from 'axios'
import qs from 'qs'
import Vue from "vue"

axios.defaults.timeout = 10000
axios.defaults.baseURL = ''
axios.defaults.withCredentials = false

//请求中的的url做保存，一旦请求完毕则移除掉
const DuplicateQueryManager = ()=>{

  var obj = new Object()
  obj._queryingURLs = {}
  obj.isQuerying = (url) => {
    return url in obj._queryingURLs
  }

  obj.add = (url) => {
    obj._queryingURLs[url] = true
  }

  obj.remove = (url) => {
    if(obj.isQuerying(url))
      delete obj._queryingURLs[url]
  }
  return obj
}

//重复请求管理器
const duplicateQueryManager = DuplicateQueryManager()
let preQueryCallback = null   //请求前的回调
let endQueryCallback = null   //请求后的回调

//请求前的调用，回调函数没有给予参数
function setPreQueryCallback(callback){
  preQueryCallback = callback
}

//请求结束后的调用，无论是否发生错误都会调用，返回的参数带有一个data参数，是服务器返回的data
function setEndQueryCallback(callback){
  endQueryCallback = callback
}

function mergeMap(target, src){
  if(target == null){
    return src
  }else if(src != null){
    for(let k in src)
      target[k] = src[k]
  }
  return target
}

function preprocessQueryParameters(param){

  if(typeof(param) == "string"){
    return param;
  }

  let omap = {}
  for(let key in param){
    let value = param[key]
    if(typeof(value) == "object"){
      value = JSON.stringify(value)
    }
    omap[key] = value
  }
  return qs.stringify(omap)
}

//定义echo对象
const Echo = (url) => {
  var query = new Object();
  query._url = url  

  /******************************公开的函数定义******************************/
  query.data = (data) =>{
    query._data = data
    return query
  }

  query.defResp = (data) =>{
    query._defResp = data
    return query
  }

  //then返回的数据已经拨离了外层，返回的是msg
  query.then = (callback) =>{
    query._then = callback
    query._send("post")
    return query
  }

  //catch返回的数据，则是完整的服务器返回数据，包括了status、msg、code等，具体错误原因都在
  query.catch = (callback) =>{
    query._catch = callback
    return query
  }

  query.config = (config) =>{
    query._config = config
    return query
  }

  query.binary_resp = () =>{
    query._binary_resp = true
    return query
  }

  //设置加载框静默，也就是不显示
  query.loadingSilence = ()=>{
    query._loadingSilence = true
    return query
  }

  //设置tips静默，也就是不显示
  query.tipsSilence = ()=>{
    query._tipsSilence = true
    return query
  }

  query.post = () =>{
    query._send("post")
    return query
  }

  query.get = () =>{
    query._send("get")
    return query
  }
  /******************************公开的函数定义******************************/

  /* 默认参数定义 */
  query._alreadyQuery = false
  query._then = (msg)=>{}
  query._catch = (msg)=>{}
  query._data = {}
  query._defResp = null
  query._binary_resp = false
  query._loadingSilence = false
  query._tipsSilence = false
  query._config = {headers: {'Content-Type': 'application/x-www-form-urlencoded', 'X-Requested-With': 'XMLHttpRequest'}}
  /*****************************/

  query._send = (type) =>{

    if(query._alreadyQuery) return
    query._alreadyQuery = true

    if(duplicateQueryManager.isQuerying(query._url)){
      console.log("重复请求了" + query._url)
      return
    }

    //如果有注册回调函数
    if(preQueryCallback != null){
      preQueryCallback(query)
    }

    //添加记录
    duplicateQueryManager.add(query._url)

    let q = null
    if(type == "post"){
      q = axios.post(query._url, preprocessQueryParameters(query._data), query._config)
    }else{
      q = axios.get(query._url, preprocessQueryParameters(query._data), query._config)
    }

    q.then((response) => {
      duplicateQueryManager.remove(query._url)

      let data = response.data
      if(query._binary_resp && response.status==200){
        query._then(data);
        if(endQueryCallback != null){
          endQueryCallback(query, null)
        }
        return;
      }

      if(endQueryCallback != null){
        endQueryCallback(query, data)
      }

      data = mergeMap(data, query._defResp)
      if(data.status == "success")
        query._then(data.data)
      else
        query._catch(data.message)
      
    }).catch((e) => {
      let response = e.response
      let dispatchData = null
      if(response == null){
        //没有收到服务器的响应
        dispatchData = {
          status: "error",
          code: "NetworkError",
          message: e.message,
          now: new Date()
        }
      }else{
        dispatchData = response.data

        //这里有两种情况，一种情况是从服务器收到的例如404、500等信号，直接使用response即可
        //另一种是没有收到服务器信号，后端服务器挂掉了，这里返回的response是前端服务器控制的或者js控制的
        //那么会没有msg属性，data属性是个string
        if(!dispatchData.hasOwnProperty("message")){
          dispatchData = {
            status: "error",
            code: response.status,
            message: response.data,
            header: response.headers,
            now: new Date()
          }
        }
      }

      dispatchData = mergeMap(dispatchData, query._defResp)
      if(endQueryCallback != null){
        endQueryCallback(query, dispatchData)
      }
      duplicateQueryManager.remove(query._url)
      query._catch(dispatchData)
    })
  }
  return query;
};

Echo.setPreQueryCallback = setPreQueryCallback
Echo.setEndQueryCallback = setEndQueryCallback

Vue.prototype.$echo = Echo
export default Echo