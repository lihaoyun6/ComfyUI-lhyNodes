import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 用来存储当前画布上所有的预览节点（允许用户建立多个）
const previewNodes = new Set();
const nodeTitles = new Set();
let initialized = false
let currentBlobUrl = null;
let currentProgress = "";

app.registerExtension({
    name: "lhyNodes.LivePreviewer",
    
    // 在节点被注册到 LiteGraph 引擎之前，修改它的行为
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LivePreviewer") {
            
            // 节点被创建时的初始化钩子
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                
                this.size = [300, 300]; // 默认大小
                this.resizable = true;
                this.imgs = []; // ComfyUI 内部通过读取 this.imgs 数组来绘制图片
                
                previewNodes.add(this);
            };

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (onRemoved) onRemoved.apply(this, arguments);
                previewNodes.delete(this);
            };
        }
    },
    
    setup() {
        // 监听实时预览二进制图片数据
        api.addEventListener("b_preview", (event) => {
            const blob = event.detail; 
            if (blob && previewNodes.size > 0) {
                // 清理旧的 URL 以防内存泄漏
                if (currentBlobUrl) {
                    URL.revokeObjectURL(currentBlobUrl);
                }
                
                currentBlobUrl = URL.createObjectURL(blob);
                
                const img = new Image();
                img.onload = () => {
                    // 更新画布上所有的预览节点
                    previewNodes.forEach((node, index) => {
                        node.imgs = [img];
                        node.title = `${currentProgress}`;
                        node.setDirtyCanvas(true, true);
                    });
                };
                img.src = currentBlobUrl;
            }
        });
        
        const clearPreview = () => {
            initialized = false
            currentProgress = "";
            if (currentBlobUrl) {
                URL.revokeObjectURL(currentBlobUrl);
                currentBlobUrl = null;
            }
            previewNodes.forEach((node, index) => {
                node.title = nodeTitles[index]
                //node.imgs = [];
                //node.setDirtyCanvas(true, true);
            });
        };
        
        api.addEventListener("progress", (event) => {
            const { value, max } = event.detail;
            if (max > 0) {
                currentProgress = `${value}/${max}`;
                previewNodes.forEach(node => node.setDirtyCanvas(true, false));
            }
        });
        
        api.addEventListener("executing", (event) => {
            if (!event.detail) {
                clearPreview();
            } else {
                if (!initialized) {
                    previewNodes.forEach((node, index) => {
                        nodeTitles[index] = node.title
                    });
                    initialized = true
                }
            }
        });
        
        api.addEventListener("execution_interrupted", () => {
            clearPreview();
        });
    }
});