import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function generateUUID() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        const v = c === "x" ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

async function uploadFilesToBatch(node, files) {
    if (!files?.length) return;
    
    const validFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
    if (!validFiles.length) {
        console.log("[BatchUpload] No valid image files.");
        return;
    }
    
    const widgets = node.widgets || [];
    const appendWidget = widgets.find(w => w.name === "append");
    const batchWidget  = widgets.find(w => w.name === "batch");
    const btn          = widgets.find(w => w.type === "button");
    
    const isAppend     = !!appendWidget?.value;
    const currentBatch = batchWidget?.value;
    
    const uuid =
    isAppend && currentBatch && currentBatch !== "None"
    ? currentBatch
    : generateUUID();
    
    const subfolder = `batch/${uuid}`;
    
    console.log(
        isAppend && currentBatch && currentBatch !== "None"
        ? `[BatchUpload] Appending to batch: ${uuid}`
        : `[BatchUpload] Creating new batch: ${uuid}`
    );
    
    const originalLabel = btn?.name;
    if (btn) btn.name = `Uploading ${validFiles.length} files...`;
    
    try {
        await Promise.all(
            validFiles.map(file => {
                const body = new FormData();
                body.append("image", file);
                body.append("subfolder", subfolder);
                body.append("overwrite", "true");
                body.append("type", "input");
                
                return api.fetchApi("/upload/image", {
                    method: "POST",
                    body,
                }).then(res => {
                    if (res.status !== 200) {
                        throw new Error(`Upload failed: ${res.status}`);
                    }
                    return res.json();
                });
            })
        );
        
        if (batchWidget) {
            if (!isAppend || currentBatch === "None") {
                const values = batchWidget.options.values;
                if (values.length === 1 && values[0] === "None") {
                    values.length = 0;
                }
                if (!values.includes(uuid)) {
                    values.unshift(uuid);
                }
                batchWidget.value = uuid;
            }
            batchWidget.callback?.(uuid);
        }
        
        if (btn) btn.name = "Generating Preview...";
        
        const res = await api.fetchApi("/batch_preview/gen_batch", {
            method: "POST",
            body: JSON.stringify({ batch_folder: uuid }),
        });
        
        if (res.status !== 200) {
            console.error("[BatchUpload] Preview generation failed:", res.statusText);
            return;
        }
        
        console.log(`[BatchUpload] Success: ${uuid}`);
        
    } catch (err) {
        console.error("[BatchUpload] Error:", err);
        alert("Upload failed: " + err.message);
    } finally {
        if (btn && originalLabel) btn.name = originalLabel;
        app.graph.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "Comfy.BatchUploadNode",
    
    nodeCreated(node) {
        if (node.comfyClass !== "LoadImageBatch") return;
        
        node.addWidget(
            "button",
            "Choose files to upload",
            "",
            () => {
                const input = document.createElement("input");
                input.type = "file";
                input.multiple = true;
                input.accept = "image/*";
                input.style.display = "none";
                
                input.onchange = async () => {
                    await uploadFilesToBatch(node, input.files);
                    input.remove();
                };
                
                document.body.appendChild(input);
                input.click();
            },
            { serialize: false }
        );
        
        node.onDragOver = function (e) {
            return !!(
                e.dataTransfer &&
                e.dataTransfer.items &&
                Array.from(e.dataTransfer.items).some(i => i.kind === "file")
            );
        };
        
        node.onDragDrop = function (e) {
            if (e.dataTransfer?.files?.length) {
                uploadFilesToBatch(node, e.dataTransfer.files);
                return true;
            }
            return false;
        };
        
        node.onPaste = function (e) {
            if (e.clipboardData?.files?.length) {
                uploadFilesToBatch(node, e.clipboardData.files);
                return true;
            }
        };
    },
});