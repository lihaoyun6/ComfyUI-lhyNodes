import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function generateUUID() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

async function fetchAndShowPreview(node, batchFolderUUID) {
    try {
        const response = await api.fetchApi("/batch_preview/gen_batch", {
            method: "POST",
            body: JSON.stringify({ batch_folder: batchFolderUUID }),
        });
        
        if (response.status !== 200) {
            console.error("Preview generation failed", response.statusText);
            return;
        }
        
        const imgInfo = await response.json();

        const previewUrl = api.apiURL(
            `/view?filename=${encodeURIComponent(imgInfo.filename)}&type=${imgInfo.type}&subfolder=${encodeURIComponent(imgInfo.subfolder)}&t=${Date.now()}`
        );

        const img = new Image();
        img.onload = () => {
            node.imgs = [img];
            app.graph.setDirtyCanvas(true, true);
        };
        img.src = previewUrl;
        
    } catch (error) {
        console.error("Error fetching preview:", error);
    }
}

async function uploadFilesToBatch(node, files) {
    if (!files || files.length === 0) return;
    
    const validFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
    if (validFiles.length === 0) {
        console.log("No valid image files found.");
        return;
    }
    
    const appendWidget = node.widgets.find(w => w.name === "append");
    const batchWidget = node.widgets.find(w => w.name === "batch");
    
    let uuid;
    const isAppend = appendWidget ? appendWidget.value : false;
    const currentBatch = batchWidget ? batchWidget.value : "None";
    
    if (isAppend && currentBatch !== "None" && currentBatch !== "") {
        uuid = currentBatch;
        console.log(`[BatchUpload] Appending files to existing batch: ${uuid}`);
    } else {
        uuid = generateUUID();
        console.log(`[BatchUpload] Creating new batch: ${uuid}`);
    }
    
    const subfolderName = "batch/" + uuid;
    const btn = node.widgets.find(w => w.type === "button");
    const originalLabel = btn.name;
    if (btn) btn.name = `Uploading ${validFiles.length} files...`;
    
    try {
        const promises = validFiles.map(async (file) => {
            const body = new FormData();
            body.append("image", file);
            body.append("subfolder", subfolderName);
            body.append("overwrite", "true");
            body.append("type", "input");
            
            const response = await api.fetchApi("/upload/image", {
                method: "POST",
                body,
            });
            if (response.status !== 200) {
                throw new Error(`Status ${response.status}`);
            }
            return response.json();
        });
        
        await Promise.all(promises);
        
        if (batchWidget) {
            if (!isAppend || currentBatch === "None") {
                if (batchWidget.options.values.length === 1 && batchWidget.options.values[0] === "None") {
                    batchWidget.options.values = [];
                }
                if (!batchWidget.options.values.includes(uuid)) {
                    batchWidget.options.values.unshift(uuid);
                }
                batchWidget.value = uuid;
            }
            if (batchWidget.callback) {
                batchWidget.callback(uuid);
            }
        }
        
        if (btn) btn.name = "Generating Preview...";
        await fetchAndShowPreview(node, uuid, true); 
        console.log(`[BatchUpload] Success: ${uuid}`);
        
    } catch (error) {
        console.error("[BatchUpload] Error:", error);
        alert("Upload failed: " + error.message);
    } finally {
        if (btn) btn.name = originalLabel;
        app.graph.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "Comfy.BatchUploadNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LoadImageBatch") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                this.addWidget("button", "Choose files to upload", "", () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.multiple = true;
                    input.accept = "image/*";
                    input.style.display = "none";
                    
                    input.onchange = async (e) => {
                        await uploadFilesToBatch(node, input.files);
                        input.remove();
                    };

                    document.body.appendChild(input);
                    input.click();
                }, {
                    serialize: false
                });
                
                node.onDragOver = function (e) {
                    if (e.dataTransfer && e.dataTransfer.items) {
                        const hasImage = Array.from(e.dataTransfer.items).some(item => item.kind === 'file');
                        if (hasImage) {
                            return true; 
                        }
                    }
                    return false;
                };

                node.onDragDrop = function (e) {
                    if (e.dataTransfer && e.dataTransfer.files) {
                        console.log("[BatchUpload] Files dropped:", e.dataTransfer.files);
                        uploadFilesToBatch(node, e.dataTransfer.files);
                        return true;
                    }
                    return false;
                };

                node.onPaste = function(e) {
                    if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
                        console.log("[BatchUpload] Clipboard pasted.");
                        uploadFilesToBatch(node, e.clipboardData.files);
                        return true;
                    }
                };

                return r;
            };
        }
    },
});