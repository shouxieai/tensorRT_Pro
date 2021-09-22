import onnx

# 下载的onnx的upsample节点是3个scales，需要修改为4个
model = onnx.load("dbface.onnx")
changed = False

for n in model.graph.node:
    if n.op_type == "Upsample":
        if len(n.attribute[1].floats) == 3:
            changed = True
            n.attribute[1].floats.insert(0, 1.0)

if changed:
    print("Change and save to dbface_cvt.onnx.")
    onnx.save(model, "dbface_cvt.onnx")
else:
    print("No need change.")