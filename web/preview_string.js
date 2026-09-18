import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "lhyNodes.PreviewStringBypass",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PreviewStringBypass") {
            
            function setupReadOnly(node) {
                const widget = node.widgets?.find((w) => w.name === "cached_text");
                if (widget && widget.inputEl) {
                    widget.inputEl.readOnly = true;
                    widget.inputEl.style.opacity = 0.6;
                }
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                setupReadOnly(this);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                setupReadOnly(this);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (message?.text) {
                    const text = Array.isArray(message.text) ? message.text.join("\n") : message.text;
                    const widget = this.widgets?.find((w) => w.name === "cached_text");
                    
                    if (widget) {
                        if (widget.value !== text) {
                            widget.value = text;
                            if (widget.inputEl) {
                                widget.inputEl.value = text;
                            }
                            app.graph.setDirtyCanvas(true, false);
                        }
                        setupReadOnly(this);
                    }
                }
            };
        }
    },
});