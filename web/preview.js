import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function updatePreview(node) {
    if (!node?.widgets) return;
    
    let previewUrl = null;
    
    if (node.comfyClass === "LoadImageBatch") {
        const widget = node.widgets.find(w => w.name === "batch");
        const uuid = widget?.value;
        
        if (uuid && uuid !== "None") {
            previewUrl = api.apiURL(
                `/view?filename=__preview__grid.webp&subfolder=batch/${uuid}&type=input&t=${Date.now()}`
            );
        }
    }
    
    const redraw = () => app.graph.setDirtyCanvas(true, true);
    
    if (!previewUrl) {
        node.imgs = [];
        redraw();
        return;
    }
    
    const img = new Image();
    img.onload = () => {
        node.imgs = [img];
        redraw();
    };
    img.onerror = () => {
        node.imgs = [];
        redraw();
    };
    img.src = previewUrl;
}

app.registerExtension({
    name: "lhyNodes.BatchPreviewWatcher",
    
    nodeCreated(node) {
        if (node.comfyClass !== "LoadImageBatch") return;
        
        const widget = node.widgets?.find(w => w.name === "batch");
        if (!widget) return;
        
        const originalCallback = widget.callback;
        widget.callback = function (value) {
            if (originalCallback) {
                originalCallback.apply(this, arguments);
            }
            updatePreview(node);
        };
        
        setTimeout(() => {
            updatePreview(node);
        }, 0);
    },
});