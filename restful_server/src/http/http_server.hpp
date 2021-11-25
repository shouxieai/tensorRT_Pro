#ifndef HTTP_SERVER_HPP
#define HTTP_SERVER_HPP

#pragma once

#include "ilogger.hpp"
#include "binary_io.hpp"
#include "json.hpp"
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>

typedef unsigned long SessionID;

enum ResponseWriteMode : int{
	ResponseWriteMode_WriteReturnJson = 0,
	ResponseWriteMode_WriteCustom = 1,
	ResponseWriteMode_WriteFile = 2
};

struct Response{
	BinaryIO output;
	std::unordered_map<std::string, std::string> headers;
	int status_code = 0;
	ResponseWriteMode write_mode = ResponseWriteMode_WriteReturnJson;
	std::string file_path;

	Response();
	void set_status_code(int code);
	void set_header(const std::string& key, const std::string& value);
	void write_binary(const void* pdata, size_t size);
	void write_json_styled(const Json::Value& val);
	void write_json(const Json::Value& val);
	void write_plain_text(const std::string& val);
	void write_file(const std::string& file);
	std::string get_header(const std::string& name);
	void remove_header(const std::string& name);
	bool has_header(const std::string& name);

	std::string header_string();
	const std::string& output_string();
};

struct Request{
	std::string url;
	std::string body;
	std::string method;
	std::string proto;
	std::string query_string;
	std::unordered_map<std::string, std::string> headers;
	std::unordered_map<std::string, std::string> vars;

	bool has_header(const std::string& name);
	std::string get_header(const std::string& name);
};

struct Session {
	SessionID conn_id; 
	Request request;
	Response response;

	Session();
	Session(SessionID id);
};

typedef std::function<void(const std::shared_ptr<Session>& session)> HandlerCallback;

#define DefRequestMapping(name)  																			\
	int __##name##_register__{add_router(#name, "*", std::bind(&__Current_Class__::name, this, std::placeholders::_1))};	\
	Json::Value name(const Json::Value& param)

#define SetupController(classes)	using __Current_Class__ = classes;

class HttpServer;
class Controller{
protected:
	typedef std::function<Json::Value(const Json::Value&)> ControllerProcess;
	struct RequestMapping{																		
		std::unordered_map<std::string, std::unordered_map<std::string, ControllerProcess>> routers;						
	}router_mapping_;

	std::string mapping_url_;
	std::unordered_map<std::thread::id, std::shared_ptr<Session>> current_session_;
	std::mutex session_lock_;

public:
	void initialize(const std::string& url, HttpServer* server);
	virtual void process(const std::shared_ptr<Session>& session);
	virtual void process_module(const std::shared_ptr<Session>& session, const ControllerProcess& func);
	virtual ControllerProcess find_match(const std::string& url, const std::string& method);
	virtual bool is_begin_match();

protected:
	int add_router(const std::string& url, const std::string& method, const ControllerProcess& process);
	std::shared_ptr<Session> get_current_session();
};

std::shared_ptr<Controller> create_redirect_access_controller(const std::string& root_directory, const std::string& root_redirect_file="");
std::shared_ptr<Controller> create_file_access_controller(const std::string& root_directory);

class HttpServer{
public:
	// method POST GET
	virtual void verbose() = 0;
	virtual void add_router(const std::string& url, const HandlerCallback& callback, const std::string& method) = 0;
	virtual void add_router_post(const std::string& url, const HandlerCallback& callback) = 0;
	virtual void add_router_get(const std::string& url, const HandlerCallback& callback) = 0;
	virtual void add_controller(const std::string& url, std::shared_ptr<Controller> controller) = 0;
};

std::shared_ptr<HttpServer> createHttpServer(const std::string& address, int num_threads = 32);

template<typename _T>
Json::Value success(const _T& value){
	Json::Value ret;
	ret["status"] = "success";
	ret["data"] = value;
	ret["time"] = iLogger::time_now();
	return ret;
}

Json::Value success();

template<typename _T>
Json::Value failure(const _T& value){
	Json::Value ret;
	ret["status"] = "error";
	ret["message"] = value;
	ret["time"] = iLogger::time_now();
	return ret;
}

Json::Value failuref(const char* fmt, ...);

#endif //HTTP_SERVER_HPP