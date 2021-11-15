import ElementUI from 'element-ui'
import Vue from 'vue'

function pack_opt(opt_or_message, title){

  let opt = {}
  if(typeof(opt_or_message) == "string"){
    opt["message"] = opt_or_message
    opt["title"] = title
  }else{
    opt = opt_or_message
  }

  if(!("duration" in opt))
    opt["duration"] = 2000
  return opt
}

function tips_success(opt_or_message, title = "成功"){
  let opt = pack_opt(opt_or_message, title)
  ElementUI.Notification.success(opt)
}

function tips_error(opt_or_message, title = "错误"){
  let opt = pack_opt(opt_or_message, title)
  ElementUI.Notification.error(opt)
}

function tips_info(opt_or_message, title = "提示"){
  let opt = pack_opt(opt_or_message, title)
  ElementUI.Notification.info(opt)
}

function tips_warning(opt_or_message, title = "警告"){
  let opt = pack_opt(opt_or_message, title)
  ElementUI.Notification.warning(opt)
}

function tips_tip(opt){
  ElementUI.Notification(opt)
}

const Tips = {
  success: tips_success,
  error: tips_error,
  info: tips_info,
  warning: tips_warning,
  tip: tips_tip
}

Vue.prototype.$tips = Tips
export default Tips