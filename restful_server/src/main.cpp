
#include "http/http_server.hpp"
#include "infer/simple_yolo.hpp"
 
using namespace std;
using namespace cv;
 
static const char* cocolabels[] = {
	"person", "bicycle", "car", "motorcycle", "airplane",
	"bus", "train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
	"laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush"
};

class InferInstance{
public:
	bool startup(){

		infer_ = get_infer(SimpleYolo::Type::X, SimpleYolo::Mode::FP32, "yolox_s");
		return infer_ != nullptr;
	}

	bool inference(const Mat& image_input, SimpleYolo::BoxArray& boxarray){
		
		if(infer_ == nullptr){
			INFOE("Not Initialize.");
			return false;
		}

		if(image_input.empty()){
			INFOE("Image is empty.");
			return false;
		}
		boxarray = infer_->commit(image_input).get();
		return true;
	}

private:
	bool requires_model(const string& name) {

		auto onnx_file = cv::format("%s_dynamic.onnx", name.c_str());
		if (!iLogger::exists(onnx_file)) {
			printf("Auto download %s\n", onnx_file.c_str());
			system(cv::format("wget http://zifuture.com:1556/fs/25.shared/%s", onnx_file.c_str()).c_str());
		}

		bool isexists = iLogger::exists(onnx_file);
		if (!isexists) {
			printf("Download %s failed\n", onnx_file.c_str());
		}
		return isexists;
	}

	shared_ptr<SimpleYolo::Infer> get_infer(SimpleYolo::Type type, SimpleYolo::Mode mode, const string& model){

		int deviceid = 1;
		auto mode_name = SimpleYolo::mode_string(mode);
		SimpleYolo::set_device(deviceid);

		const char* name = model.c_str();
		printf("===================== test %s %s %s ==================================\n", SimpleYolo::type_name(type), mode_name, name);

		if(!requires_model(name))
			return nullptr;

		string onnx_file = cv::format("%s_dynamic.onnx", name);
		string model_file = cv::format("%s_dynamic.%s.trtmodel", name, mode_name);
		int test_batch_size = 16;
		
		if(!iLogger::exists(model_file)){
			SimpleYolo::compile(
				mode, type,                 // FP32、FP16、INT8
				test_batch_size,            // max batch size
				onnx_file,                  // source 
				model_file,                 // save to
				1 << 30,
				"inference"
			);
		}
		return SimpleYolo::create_infer(model_file, type, deviceid, 0.25f, 0.5f);
	}
	
private:
	shared_ptr<SimpleYolo::Infer> infer_;
};

class LogicalController : public Controller{
	SetupController(LogicalController);

public:
	bool startup();
 
public: 
	DefRequestMapping(getCustom);
	DefRequestMapping(getReturn);
	DefRequestMapping(getBinary);
	DefRequestMapping(getFile);
	DefRequestMapping(putBase64Image);
	DefRequestMapping(detectBase64Image);

private:
	shared_ptr<InferInstance> infer_instance_;
};

Json::Value LogicalController::putBase64Image(const Json::Value& param){

	/**
	 * 注意，这个函数的调用，请用工具（postman）以提交body的方式(raw)提交base64数据
	 * 才能够在request.body中拿到对应的base64，并正确解码后储存
	 * 1. 可以在网页上提交一个图片文件，并使用postman进行body-raw提交，例如网址是：https://base64.us/，选择页面下面的“选择文件”按钮
	 * 2. 去掉生成的base64数据前缀：data:image/png;base64,。保证是纯base64数据输入
	 *   这是一个图像的base64案例：iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsQAAA7EAZUrDhsAAABLSURBVEhLY2RY9OI/Ax0BE5SmG6DIh/8DJKAswoBxwwswTXcfjlpIdTBqIdXBqIVUB8O/8B61kOpg1EKqg1ELqQ5GLaQ6oLOFDAwA5z0K0dyTzgcAAAAASUVORK5CYII=
	 *   提交后能看到是个天蓝色的背景加上右上角有黄色的正方形
	 */

	auto session = get_current_session();
	auto image_data = iLogger::base64_decode(session->request.body);
	iLogger::save_file("base_decode.jpg", image_data);
	return success();
}

Json::Value LogicalController::detectBase64Image(const Json::Value& param){

	auto session = get_current_session();
	auto image_data = iLogger::base64_decode(session->request.body);
	if(image_data.empty())
		return failure("Image is required");

	auto image = cv::imdecode(image_data, 1);
	if(image_data.empty())
		return failure("Image is empty");

	SimpleYolo::BoxArray boxarray;
	if(!this->infer_instance_->inference(image, boxarray))
		return failure("Server error1");
	
	Json::Value boxarray_json(Json::arrayValue);
	for(auto& box : boxarray){
		Json::Value item(Json::objectValue);
		item["left"] = box.left;
		item["top"] = box.top;
		item["right"] = box.right;
		item["bottom"] = box.bottom;
		item["confidence"] = box.confidence;
		item["class_label"] = box.class_label;
		item["class_name"] = cocolabels[box.class_label];
		boxarray_json.append(item);
	}
	return success(boxarray_json);
}

Json::Value LogicalController::getCustom(const Json::Value& param){

	auto session = get_current_session();
	const char* output = "hello http server";
	session->response.write_binary(output, strlen(output));
	session->response.set_header("Content-Type", "text/plain");
	return success();
}

Json::Value LogicalController::getReturn(const Json::Value& param){

	Json::Value data;
	data["alpha"] = 3.15;
	data["beta"] = 2.56;
	data["name"] = "张三";
	return success(data);
}

Json::Value LogicalController::getBinary(const Json::Value& param){

	auto session = get_current_session();
	auto data = iLogger::load_file("img.jpg");
	session->response.write_binary(data.data(), data.size());
	session->response.set_header("Content-Type", "image/jpeg");
	return success();
}

Json::Value LogicalController::getFile(const Json::Value& param){

	auto session = get_current_session();
	session->response.write_file("img.jpg");
	return success();
}

bool LogicalController::startup(){

	infer_instance_.reset(new InferInstance());
	if(!infer_instance_->startup()){
		infer_instance_.reset();
	}
	return infer_instance_ != nullptr;
}

int test_http(int port = 8090){

	INFO("Create controller");
	auto logical_controller = make_shared<LogicalController>();
	if(!logical_controller->startup()){
		INFOE("Startup controller failed.");
		return -1;
	}

	string address = iLogger::format("0.0.0.0:%d", port);
	INFO("Create http server to: %s", address.c_str());

	auto server = createHttpServer(address, 32);
	if(!server)
		return -1;
 
	INFO("Add controller");
	server->add_controller("/api", logical_controller);
	server->add_controller("/", create_redirect_access_controller("./web"));
	server->add_controller("/static", create_file_access_controller("./"));
	INFO("Access url: http://%s", address.c_str());

	INFO(
		"\n"
		"访问如下地址即可看到效果:\n"
		"1. http://%s/api/getCustom              使用自定义写出内容作为response\n"
		"2. http://%s/api/getReturn              使用函数返回值中的json作为response\n"
		"3. http://%s/api/getBinary              使用自定义写出二进制数据作为response\n"
		"4. http://%s/api/getFile                使用自定义写出文件路径作为response\n"
		"5. http://%s/api/putBase64Image         通过提交base64图像数据进行解码后储存\n"
		"6. http://%s/static/img.jpg             直接访问静态文件处理的controller，具体请看函数说明\n"
		"7. http://%s                            访问web页面，vue开发的",
		address.c_str(), address.c_str(), address.c_str(), address.c_str(), address.c_str(), address.c_str(), address.c_str()
	);

	INFO("按下Ctrl + C结束程序");
	return iLogger::while_loop();
}

int main(int argc, char** argv) {
	return test_http();
}