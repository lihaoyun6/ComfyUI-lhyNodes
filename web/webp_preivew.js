import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const postContinue = (nodeId) => fetch("/lhy_queuehandler/continue/" + nodeId, { method: "POST" });

app.registerExtension({
    name: "lhyNodes.LatentPreviewTinyVAE",

    init() {
        // ============================================================
        // 【核心】：监听 Python 在 return 之前通过 send_sync 发来的事件
        // ============================================================
        api.addEventListener("tiny_vae_preview", (event) => {
            const { node_id, filename, subfolder, type } = event.detail;
            
            // 根据 node_id 准确找到对应的节点
            const node = app.graph.getNodeById(node_id);

            if (node && node.previewImgEl) {
                // 同步右键菜单元数据
                node.images = [{ filename, subfolder, type, format: "webp" }];

                // 请求图片 URL，原生 HTML <img> 会立刻自动播放 WebP
                const src = api.apiURL(
                    `/view?filename=${encodeURIComponent(filename)}` +
                    `&subfolder=${encodeURIComponent(subfolder || "")}` +
                    `&type=${encodeURIComponent(type || "temp")}` +
                    `&t=${Date.now()}`
                );

                node.previewImgEl.src = src;
                node.previewImgEl.style.display = "block";
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LatentPreviewTinyVAE") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            const node = this;

            // 1. 创建 DOM 容器
            const container = document.createElement("div");
            container.style.width = "100%";
            container.style.display = "flex";
            container.style.justifyContent = "center";
            container.style.alignItems = "center";
            container.style.overflow = "visible";
            //container.style.paddingBottom = "10px";

            // 2. 创建原生的 <img> 标签
            const imgEl = document.createElement("img");
            imgEl.style.width = "100%";
            imgEl.style.height = "100%";
            imgEl.style.objectFit = "contain";
            imgEl.style.display = "none";
            imgEl.draggable = false;

            // 方便在 api.addEventListener 里通过 node.previewImgEl 快速访问
            node.previewImgEl = imgEl;
            container.appendChild(imgEl);

            // 辅助函数：更新容器高度
            const updateContainerHeight = () => {
                if (node.size) {
                    const topOffset = container.offsetTop || 100;
                    
                    // 容器可用高度 = 节点现有高度 - 顶部控件高 - 底部按钮高
                    const availableHeight = Math.max(node.size[1] - topOffset - 40, 40);
                    container.style.height = `${availableHeight}px`;
                }
            };

            // 3. 监听节点拖拽缩放
            const origOnResize = node.onResize;
            node.onResize = function (size) {
                origOnResize?.apply(this, arguments);
                updateContainerHeight();
            };

            // 4. 图片加载后，根据宽高比自动伸展节点高度
            imgEl.onload = function () {
                updateContainerHeight();
                node.setDirtyCanvas(true, true);
            };

            // 5. 挂载为 DOM Widget
            node.addDOMWidget("web_preview", "custom_preview_widget", container, {
                serialize: false,
            });
            
            node.addWidget("button", "Continue", "CONTINUE", () => {
                postContinue(node.id);
            });
        };
    },
});