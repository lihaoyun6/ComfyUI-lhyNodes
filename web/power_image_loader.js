import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const overlay = document.createElement("div");
overlay.id = "bottom-drop-port";
overlay.style.cssText = `
    display: none; 
    position: fixed;
    bottom: 0; left: 0; 
    width: 100vw; height: 33.3vh;
    background: rgba(0,0,0,0.6);
    z-index: 100000;
    pointer-events: none;
    display: flex;
    flex-direction: row;
    gap: 15px;
    padding: 20px;
    box-sizing: border-box;
    backdrop-filter: blur(2px);
    border-top: 2px solid #4CAF50;
    box-shadow: 0 -10px 30px rgba(0,0,0,0.5);
    transition: transform 0.3s ease-out;
    transform: translateY(100%);
`;

document.body.appendChild(overlay);

let dragCounter = 0;
let targetNodes = [];

function updateHorizontalLayout(nodes) {
    overlay.innerHTML = "";
    
    targetNodes = [...nodes].sort((a, b) => {
        const tA = (a.title || a.comfyClass).toLowerCase();
        const tB = (b.title || b.comfyClass).toLowerCase();
        return tA === tB ? a.id - b.id : tA.localeCompare(tB);
    });
    
    targetNodes.forEach((node, index) => {
        const isBatch = node.comfyClass === "LoadImageBatch";
        const cell = document.createElement("div");
        cell.className = "drop-cell";
        cell.style.cssText = `
            flex: 1;
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: white;
            transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            background: rgba(255, 255, 255, 0.2);
            overflow: hidden;
        `;

        const nodeTitle = node.title || node.type;
        const nodeId = node.id;
        cell.innerHTML = `
            <div style="font-size: 30px; margin-bottom: 5px;">${isBatch ? '📚' : '🖼️'}</div>
            <div style="font-size: 1.1rem; font-weight: bold; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 90%;">
                ${nodeTitle}
            </div>
            <div style="margin-top: 5px; padding: 2px 8px; background: rgba(76, 175, 80, 0.2); border-radius: 14px; font-size: 1rem; color: #4CAF50; font-family: monospace;">
                #${nodeId}
            </div>
        `;
        overlay.appendChild(cell);
    });

    overlay.style.display = "flex";
    setTimeout(() => {
        overlay.style.transform = "translateY(0)";
    }, 10);
}

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

async function uploadSingle(file, node) {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("type", "input");
    formData.append("overwrite", "true");
    const res = await api.fetchApi("/upload/image", { method: "POST", body: formData });
    if (res.ok) {
        const data = await res.json();
        const w = node.widgets.find(w => w.name === "image");
        if (w) { w.value = data.name; w.callback?.(data.name); }
    }
}

async function uploadBatch(files, node) {
    const validFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
    if (!validFiles.length) return;
    
    const widgets = node.widgets || [];
    const appendWidget = widgets.find(w => w.name === "append");
    const batchWidget = widgets.find(w => w.name === "batch");
    
    const isAppend = !!appendWidget?.value;
    const currentBatch = batchWidget?.value;
    const uuid = (isAppend && currentBatch && currentBatch !== "None") ? currentBatch : generateUUID();
    const subfolder = `batch/${uuid}`;
    
    try {
        await Promise.all(validFiles.map(file => {
            const body = new FormData();
            body.append("image", file);
            body.append("subfolder", subfolder);
            body.append("overwrite", "true");
            body.append("type", "input");
            return api.fetchApi("/upload/image", { method: "POST", body });
        }));
        
        if (batchWidget) {
            const values = batchWidget.options.values;
            if (!values.includes(uuid)) values.unshift(uuid);
            batchWidget.value = uuid;
            batchWidget.callback?.(uuid);
        }
        
        await api.fetchApi("/batch_preview/gen_batch", {
            method: "POST",
            body: JSON.stringify({ batch_folder: uuid }),
        }).catch(e => console.log("Preview service not found, skipping."));
        
    } catch (err) {
        console.error("Batch Upload Error:", err);
    }
}

function hideOverlay() {
    dragCounter = 0; 
    overlay.style.transform = "translateY(100%)";
    overlay.style.pointerEvents = "none";
    setTimeout(() => {
        if (dragCounter <= 0) {
            overlay.style.display = "none";
            overlay.innerHTML = "";
        }
    }, 400);
}

app.registerExtension({
    name: "lhyNodes.PowerImageLoader",
    async setup() {
        
        window.addEventListener("dragenter", (e) => {
            if (!e.dataTransfer.types.includes("Files")) return;
            const targetTypes = ["LoadImage", "LoadImageMask", "LoadImageOutput", "LoadImageBatch"];
            
            const nodes = app.graph._nodes.filter(node =>
                targetTypes.includes(node.type)
            );
            if (nodes.length === 0) return;

            e.preventDefault();
            dragCounter++;
            if (dragCounter === 1) {
                updateHorizontalLayout(nodes);
                overlay.style.pointerEvents = "auto"; 
            }
        }, true);

        window.addEventListener("dragover", (e) => {
            if (overlay.style.display === "none") return;
            
            e.preventDefault();
            e.stopPropagation();

            if (e.clientY > window.innerHeight * 0.66) {
                e.stopImmediatePropagation();
            }

            const cells = overlay.querySelectorAll(".drop-cell");
            cells.forEach(cell => {
                const rect = cell.getBoundingClientRect();
                if (e.clientX >= rect.left && e.clientX <= rect.right &&
                    e.clientY >= rect.top && e.clientY <= rect.bottom) {
                    cell.style.borderColor = "#4CAF50";
                    cell.style.background = "rgba(76, 175, 80, 0.25)";
                    cell.style.transform = "translateY(-5px) scale(1.02)";
                    cell.style.boxShadow = "0 10px 20px rgba(0,0,0,0.3)";
                } else {
                    cell.style.borderColor = "rgba(255, 255, 255, 0.3)";
                    cell.style.background = "rgba(255, 255, 255, 0.05)";
                    cell.style.transform = "translateY(0) scale(1)";
                    cell.style.boxShadow = "none";
                }
            });
        }, true);

        window.addEventListener("dragleave", (e) => {
            e.preventDefault();
            
            if (!e.relatedTarget || 
                e.clientX <= 0 || 
                e.clientY <= 0 || 
                e.clientX >= window.innerWidth || 
                e.clientY >= window.innerHeight) {
                    
                    hideOverlay();
                }
        }, true);

        window.addEventListener("drop", async (e) => {
            const isBottom = e.clientY > window.innerHeight * 0.66;
            const files = e.dataTransfer.files;
            
            if (overlay.style.display !== "none" && isBottom) {
                e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
                
                const cells = overlay.querySelectorAll(".drop-cell");
                let targetIndex = -1;
                cells.forEach((c, i) => {
                    const r = c.getBoundingClientRect();
                    if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) targetIndex = i;
                });
                
                if (targetIndex !== -1) {
                    const node = targetNodes[targetIndex];
                    if (node.comfyClass === "LoadImageBatch") {
                        await uploadBatch(files, node);
                    } else {
                        await uploadSingle(files[0], node);
                    }
                }
            }
            dragCounter = 0;
            hideOverlay();
        }, true);

        window.addEventListener("keydown", (e) => {
            if (e.key === "Escape") hideOverlay();
        });
    }
});