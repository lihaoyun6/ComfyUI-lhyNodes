import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const postContinue = (nodeId) => fetch("/lhy_queuehandler/continue/" + nodeId, { method: "POST" });

app.registerExtension({
    name: "lhyNodes.LatentPreviewTinyVAE",

    init() {
        api.addEventListener("tiny_vae_preview", (event) => {
            const { node_id, filename, subfolder, type } = event.detail;
            const node = app.graph.getNodeById(node_id);

            if (node && node._previewMedia) {
                const isVideo = filename.endsWith(".mp4") || filename.endsWith(".webm");
                node.images = [{ filename, subfolder, type, format: isVideo ? "mp4" : "webp" }];

                const src = api.apiURL(
                    `/view?filename=${encodeURIComponent(filename)}` +
                    `&subfolder=${encodeURIComponent(subfolder || "")}` +
                    `&type=${encodeURIComponent(type || "temp")}` +
                    `&t=${Date.now()}`
                );

                const { imgEl, videoEl } = node._previewMedia;

                if (isVideo) {
                    // MP4 视频模式：开启 Video 播放器
                    imgEl.style.display = "none";
                    videoEl.src = src;
                    videoEl.style.display = "block";
                    videoEl.muted = true;
                    videoEl.play().catch(() => {});
                } else {
                    // WebP 降级模式：切回 Image 标签
                    videoEl.pause();
                    videoEl.style.display = "none";
                    videoEl.removeAttribute("src");
                    videoEl.load();

                    imgEl.src = src;
                    imgEl.style.display = "block";
                }
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
            container.style.height = "100%";
            container.style.maxHeight = "100%";
            container.style.display = "flex";
            container.style.justifyContent = "center";
            container.style.alignItems = "center";
            container.style.overflow = "hidden";
            container.style.borderRadius = "4px";

            // 2. 创建 <img> 标签（WebP 降级通道）
            const imgEl = document.createElement("img");
            imgEl.style.width = "100%";
            imgEl.style.height = "100%";
            imgEl.style.objectFit = "contain";
            imgEl.style.display = "none";
            imgEl.draggable = false;

            // 3. 创建 <video> 标签（MP4 + 音频通道）
            const videoEl = document.createElement("video");
            videoEl.style.width = "100%";
            videoEl.style.height = "100%";
            videoEl.style.objectFit = "contain";
            videoEl.style.display = "none";
            videoEl.autoplay = true;
            videoEl.loop = true;
            videoEl.controls = true;

            node._previewMedia = { imgEl, videoEl };
            container.appendChild(imgEl);
            container.appendChild(videoEl);

            // 4. 挂载为 DOM Widget
            node.addDOMWidget("web_preview", "custom_preview_widget", container, {
                serialize: false,
            });

            // 5. 添加 Continue 按钮
            node.addWidget("button", "Continue", "CONTINUE", () => {
                postContinue(node.id);
            });
        };
    },
});