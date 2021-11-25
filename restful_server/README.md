# 基于Yolo推理的Resful API http server
- C++的http服务器，简单好用
![](web/preview.jpg)

## 执行方式
```bash
make run -j6
```

## HTTP服务的案例代码
```C++
#include "http_server.hpp"

using namespace std;

// 定义控制器，处理路由对应的请求，MVC模式里边的C部分，
class LogicalController : public Controller{
    SetupController(LogicalController);
 
public: 
    // 轻易请求的映射，这里的名称就是路由的地址
    DefRequestMapping(getReturn);
};

// 定义路由的处理函数，类java spring。这里使用返回值作为response。实际上支持多种，请看具体代码
Json::Value LogicalController::getReturn(const Json::Value& param){

    Json::Value data;
    data["alpha"] = 3.15;
    data["name"] = "张三";
    return success(data);
}

int main(int argc, char** argv) {

    // 创建http服务器，具有32个线程
    auto server = createHttpServer("0.0.0.0:8090", 32);
    if(server == nullptr) return -1;
    
    // 添加控制器对于路由v1
    server->add_controller("/v1", make_shared<LogicalController>());

    // 添加静态文件访问，对于路由/static，绑定到路径./上
    server->add_controller("/static", create_file_access_controller("./"));
    return while_loop();
}
```

## 效果
![](workspace/demo1.jpg)
![](workspace/demo2.jpg)

## 关于我们
- 我们的博客是：http://www.zifuture.com:8090/
- 我们的B站地址是：https://space.bilibili.com/1413433465