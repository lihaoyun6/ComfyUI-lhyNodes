import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const previewNodes = new Set();
const nodeTitles = new Set();
let initialized = false
let currentBlobUrl = null;
let currentProgress = "";

app.registerExtension({
    name: "lhyNodes.LivePreviewer",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LivePreviewer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                
                this.size = [300, 300];
                this.resizable = true;
                this.imgs = [];
                
                previewNodes.add(this);
            };

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (onRemoved) onRemoved.apply(this, arguments);
                previewNodes.delete(this);
            };
        }
        
        if (nodeData.name === "DynamicParameterPanel") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                previewNodes.add(this);
                return r;
            };
            
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                const r = onRemoved?.apply(this, arguments);
                previewNodes.delete(this);
                return r;
            };
        }
    },
    
    setup() {
        api.addEventListener("b_preview", (event) => {
            const blob = event.detail; 
            if (blob && previewNodes.size > 0) {
                if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
                
                currentBlobUrl = URL.createObjectURL(blob);
                
                const img = new Image();
                img.onload = () => {
                    previewNodes.forEach((node, index) => {
                        if (node.type === "DynamicParameterPanel") {
                            const pw = node.widgets.find(w => w.name === "live_preview");
                            const enabled = pw?.value === true;
                            if (!enabled) return;
                        }
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