import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function loadHeic2Any() {
    if (window.heic2any) return Promise.resolve(window.heic2any);
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = new URL('./heic2any/heic2any.min.js', import.meta.url).href;
        script.onload = () => resolve(window.heic2any);
        script.onerror = () => {
            console.error("[lhyNodes] Failed to load heic2any script!");
            reject(new Error("HEIC decoding library failed to load!"));
        };
        document.head.appendChild(script);
    });
}

function convertAvifToPng(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            canvas.toBlob((blob) => {
                URL.revokeObjectURL(url);
                if (blob) {
                    const newName = file.name.replace(/\.avif$/i, '.png');
                    resolve(new File([blob], newName, { type: 'image/png' }));
                } else {
                    console.error("[lhyNodes] Failed to export PNG from canvas!");
                    reject(new Error("Failed to export PNG!"));
                }
            }, 'image/png');
        };
        img.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error("Your browser doesn't support the AVIF format!"));
        };
        img.src = url;
    });
}

app.registerExtension({
    name: "lhyNodes.FrontendFormatConverter",
    async setup() {
        const originalFetchApi = api.fetchApi;

        api.fetchApi = async function(route, options) {
            if ((route === "/upload/image" || route === "/upload/mask") && options && options.body instanceof FormData) {
                const file = options.body.get("image");
                if (file && file instanceof File) {
                    const ext = file.name.split('.').pop().toLowerCase();
                    
                    try {
                        if (ext === 'heic' || ext === 'heif') {
                            console.log("[lhyNodes] Processing HEIC/HEIF image...");
                            const h2a = await loadHeic2Any();
                            const blobResult = await h2a({ blob: file, toType: "image/png" });
                            const blob = Array.isArray(blobResult) ? blobResult[0] : blobResult;
                            const newFile = new File([blob], file.name.replace(/\.hei[cf]$/i, '.png'), { type: 'image/png' });
                            options.body.set("image", newFile);
                        } 
                        else if (ext === 'avif') {
                            console.log("[lhyNodes] Processing AVIF image...");
                            const newFile = await convertAvifToPng(file);
                            options.body.set("image", newFile);
                        }
                    } catch (e) {
                        console.error("[lhyNodes] Unable to convert image: ", e);
                        alert("Unable to convert image: " + e.message);
                    }
                }
            }
            return originalFetchApi.apply(this, arguments);
        };

        const originalClick = HTMLInputElement.prototype.click;
        HTMLInputElement.prototype.click = function() {
            if (this.type === 'file' && typeof this.accept === 'string') {
                const acceptStr = this.accept.toLowerCase();
                
                const hasImage = acceptStr.includes("image/") || acceptStr.includes(".jpg") || acceptStr.includes(".png");
                const hasVideo = acceptStr.includes("video/") || acceptStr.includes(".mp4") || acceptStr.includes(".webm");
                const hasAudio = acceptStr.includes("audio/");
                
                if (hasImage && !hasVideo && !hasAudio) {
                    if (!acceptStr.includes(".heic")) {
                        this.accept += ",.heic,.heif,.avif,image/heic,image/heif,image/avif";
                    }
                }
            }
            return originalClick.apply(this, arguments);
        };
    }
});