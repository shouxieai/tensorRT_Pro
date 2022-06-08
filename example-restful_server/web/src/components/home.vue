<template>
    <div class="v-body">
        <input type="file" style="display: none" ref="uploadImage" @change="fileChange" accept="image/jpeg, image/png" multiple />
        <el-container>
            <el-aside width="800px">
                <canvas width="800px" height="700px" ref="previewImageCanvas" class="detect-image"></canvas>
            </el-aside>
            <el-main>
                <el-button type="primary" @click="dotest">上传图像测试</el-button>
                <p></p>
                <el-input type="textarea" :rows="32" placeholder="检测结果" v-model="detectResultJson" class="detect-result-json" :readonly="true">
                </el-input>
            </el-main>
        </el-container>
    </div>
</template>
<style scoped>
.v-body{
    padding: 100px;
}
</style>
<script>
export default {
    name: 'Home',
    data(){
        return {
            uploadFileSelectCallback: null,
            detectResultJson: ""
        }
    },
    methods: {
        fileChange: function(){
            let files = this.$refs.uploadImage.files;
            if(files.length != 1){
                this.$message.warning("请选择一个文件");
                return;
            }

            if(this.uploadFileSelectCallback)
                this.uploadFileSelectCallback(files[0]);

            this.$refs.uploadImage.value = null;
        },
        openFileDialog: function(callback){
            this.uploadFileSelectCallback = callback;
            this.$refs.uploadImage.click();
        },
        loadImage: function(url, callback){
            let image = new Image();
            image.onload = function(){
                callback(image);
            };
            image.src = url;
        },
        drawImageAndLabel: function(image, boxarray){

            this.detectResultJson = JSON.stringify(boxarray, undefined, 4);
            let cvs = this.$refs.previewImageCanvas;
            let context = cvs.getContext("2d");
            let imageWidth = image.width;
            let imageHeight = image.height;
            let canWidth = cvs.width;
            let canHeight = cvs.height;
            let scale = Math.min(canWidth / imageWidth, canHeight / imageHeight);
            scale = Math.min(scale, 1);

            let dx = (canWidth - imageWidth * scale) / 2;
            let dy = (canHeight - imageHeight * scale) / 2;
            let dr = canWidth - dx;
            let db = canHeight - dy;
            context.clearRect(0, 0, canWidth, canHeight);
            context.drawImage(image, dx, dy, dr - dx + 1, db - dy + 1);

            if(boxarray == null || boxarray.length == 0)
                return;

            context.font = "26px 微软雅黑";
            context.save();
            context.translate(dx, dy);
            context.scale(scale, scale);

            var colors = ["#F0F", "#0FF", "#FF0", "#00F", "#0F0", "#F00", "#8F0", "#08F", "#0F8", "#88F", "#F80"];
            for(let i = 0; i < boxarray.length; ++i){
                let box = boxarray[i];
                let color = colors[box.class_label % colors.length];
                let label = box.class_name;

                context.beginPath();
                context.lineWidth = "5";
                context.strokeStyle = color;
                context.rect(
                    box.left, 
                    box.top, 
                    (box.right - box.left + 1), 
                    (box.bottom - box.top + 1));
                context.stroke();

                context.fillStyle = color;
                context.font = "25px 微软雅黑";
                context.fillText(label + " " + box.confidence.toFixed(2), box.left, box.top - 5);
            }
            context.restore();
        },
        dotest: function(){
            var _this = this;
            this.openFileDialog(function(file){
                var reader = new FileReader();
                reader.onload = function(e){ 
                    _this.loadImage(this.result, function(image){
                        
                        var token = ";base64,";
                        var base64_data = image.src;
                        var token_begin = base64_data.indexOf(token);
                        var clean_data = base64_data.substring(token_begin + token.length);
                        Echo("/api/detectBase64Image")
                        .data(clean_data)
                        .then((boxarray)=>{
                            _this.drawImageAndLabel(image, boxarray);
                        });
                    });
                };
                reader.readAsDataURL(file);
            })
        },
        init: function(){
        }
    },
    mounted(){
        this.init()
    }
}
</script>