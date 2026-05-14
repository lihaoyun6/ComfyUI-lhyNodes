import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "lhyNodes.DynamicParameterPanel",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DynamicParameterPanel") {
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                this.isConfigured = true; 
                
                if (onConfigure) onConfigure.apply(this, arguments);
                if (info && info.widgets_values) {
                    const savedJson = info.widgets_values[0];
                    this.buildDynamicUI(savedJson, true); 
                    for (let i = 0; i < info.widgets_values.length; i++) {
                        if (this.widgets[i]) this.widgets[i].value = info.widgets_values[i];
                    }
                    this.refreshLockState();
                }
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                //this.outputs = [];
                this.size = [300, 300];

                this.addWidget("button", "🔄 Update This Panel", "btn_update", () => {
                    this.buildDynamicUI(null, false);
                });
                
                this.addWidget("button", "🔒 Lock Configuration", "btn_lock", () => {
                    const lw = this.widgets.find(w => w.name === "is_locked");
                    if (lw) {
                        lw.value = !lw.value;
                        this.refreshLockState();
                    }
                });
                
                this.hideWidget = function(widget) {
                    if (!widget._origType) widget._origType = widget.type;
                    widget.hidden = true;
                    widget.type = "converted-widget";
                    widget.computeSize = () => [0, -4];
                }
                
                this.showWidget = function(widget) {
                    widget.hidden = false;
                    widget.computeSize = undefined;
                    if (widget._origType) widget.type = widget._origType;
                }

                this.refreshLockState = function() {
                    const lw = this.widgets.find(w => w.name === "is_locked");
                    if (!lw) return;
                    const isLocked = lw.value;
                    this.hideWidget(lw);

                    const jsonW = this.widgets.find(w => w.name === "config_json");
                    const lockBtn = this.widgets.find(w => w.value === "btn_lock");
                    const updateBtn = this.widgets.find(w => w.value === "btn_update");
                    
                    if (isLocked) {
                        this.hideWidget(jsonW);
                        this.hideWidget(updateBtn);
                        if (lockBtn) lockBtn.name = "🔓 Unlock Configuration";
                    } else {
                        this.showWidget(jsonW);
                        this.showWidget(updateBtn);
                        if (lockBtn) lockBtn.name = "🔒 Lock Configuration";
                    }
                    
                    //this.onResize?.(this.size);
                    this.computeSize();
                    this.setDirtyCanvas(true, true);
                };
                
                this.notifyUnpackers = function() {
                    if (!this.outputs || !this.outputs[0].links) return;
                    this.outputs[0].links.forEach(linkId => {
                        const link = app.graph.links[linkId];
                        if (link) {
                            const targetNode = app.graph.getNodeById(link.target_id);
                            if (targetNode && targetNode.type === "ParameterUnpacker") {
                                targetNode.syncFromUpstream();
                            }
                        }
                    });
                };

                this.buildDynamicUI = async function(customJsonStr = null, isRestoring = false) {
                    let jsonStr = customJsonStr || this.widgets.find(w => w.name === "config_json")?.value;
                    if (jsonStr === "") jsonStr = "{}";
                    if (!jsonStr) return;

                    let config;
                    try { config = JSON.parse(jsonStr); } catch (e) {
                        if (!isRestoring) alert("Invalid JSON!\n" + e.message); return;
                    }
                    
                    const entries = Object.entries(config).slice(0, 32);

                    const isStatic = (w) => {
                        return w.name === "config_json" || w.name === "is_locked" || w.value === "btn_update" || w.value === "btn_lock";
                    };

                    const currentValues = {};
                    if (!isRestoring) {
                        this.widgets.forEach(w => { if(!isStatic(w)) currentValues[w.name] = w.value; });
                    }

                    for (let i = this.widgets.length - 1; i >= 0; i--) {
                        if (!isStatic(this.widgets[i])) {
                            if (this.widgets[i].onRemove) this.widgets[i].onRemove();
                            this.widgets.splice(i, 1);
                        }
                    }

                    for (const [key, params] of entries) {
                        const type = (params.type || "STRING").toUpperCase();
                        let widget;
                        let val = params.default;
                        //let val = currentValues[key] ?? params.default;

                        if (type === "INT" || type === "FLOAT") {
                            const isInt = type === "INT";
                            const step = params.step ?? (isInt ? 1 : 0.1);
                            const prec = params.precision ?? (isInt ? 0 : 3);
                            const isSlider = params.display === "slider";
                            
                            widget = this.addWidget(isSlider ? "slider" : "number", key, val || 0, (v) => {
                                let snapped;
                                if (isInt) {
                                    snapped = Math.round(v);
                                } else {
                                    snapped = Math.round(v / step) * step;
                                }
                                
                                const finalVal = parseFloat(snapped.toFixed(prec));
                                if (widget.value !== finalVal) {
                                    widget.value = finalVal;
                                }
                            }, { 
                                min: params.min ?? -999999, 
                                max: params.max ?? 999999, 
                                step: isSlider ? step : step * 10, 
                                precision: prec 
                            });
                        } else if (type === "STRING") {
                            if (params.multiline || params.placeholder) {
                                ComfyWidgets.STRING(this, key, ["STRING", { multiline: !!params.multiline, default: val || "" }], app);
                                widget = this.widgets[this.widgets.length - 1];
                                widget.value = val || "";
                                if (widget.inputEl && params.placeholder) widget.inputEl.placeholder = params.placeholder;
                            } else {
                                widget = this.addWidget("text", key, val || "", null, {});
                            }
                        } else if (type === "COMBO") {
                            let values = params.values || ["None"];
                            
                            if (params.folder) {
                                try {
                                    const response = await fetch(`/dynamic_panel/files/${params.folder}`);
                                    if (response.ok) {
                                        values = await response.json();
                                    }
                                } catch (e) { console.error("Fetch models failed:", e); }
                            }
                            
                            widget = this.addWidget("combo", key, val || values[0], null, { values: values });
                        } else if (type === "BOOLEAN") {
                            widget = this.addWidget("toggle", key, val ?? true, null, {});
                        }
                        if (widget) { widget.label = params.name || key; widget.tooltip = params.tooltip; }
                    }
                    
                    if (!isRestoring) this.notifyUnpackers();
                    this.refreshLockState();
                };

                setTimeout(() => {
                    if (!this.isConfigured) {
                        this.buildDynamicUI(null, true); 
                    } else {
                        this.refreshLockState();
                    }
                }, 10);
                return r;
            };
        }
        
        if (nodeData.name === "ParameterUnpacker") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.outputs = [];
                this.size = [200, 40];
                
                this.syncFromUpstream = function() {
                    if (!this.inputs || !this.inputs[0].link) {
                        if (this.outputs) {
                            for (let i = 0; i < this.outputs.length; i++) this.disconnectOutput(i);
                        }
                        this.outputs = [];
                        this.computeSize();
                        this.setDirtyCanvas(true, true);
                        return;
                    }
                    
                    const linkId = this.inputs[0].link;
                    const link = app.graph.links[linkId];
                    if (!link) return;
                    
                    const upstreamNode = app.graph.getNodeById(link.origin_id);
                    if (!upstreamNode || upstreamNode.type !== "DynamicParameterPanel") return;
                    
                    const jsonWidget = upstreamNode.widgets.find(w => w.name === "config_json");
                    if (!jsonWidget) return;
                    const value = jsonWidget.value || "{}"
                    
                    let config;
                    try { config = JSON.parse(value); } catch (e) { return; }
                    const entries = Object.entries(config).slice(0, 32);
                    
                    const oldLinks = [];
                    if (this.outputs) {
                        for (let i = 0; i < this.outputs.length; i++) {
                            const output = this.outputs[i];
                            if (output.links && output.links.length > 0) {
                                const linksInfo = output.links.map(lId => {
                                    const l = app.graph.links[lId];
                                    return l ? { target_id: l.target_id, target_slot: l.target_slot } : null;
                                }).filter(l => l);
                                oldLinks.push({ name: output.name, connections: linksInfo });
                                this.disconnectOutput(i);
                            }
                        }
                    }
                    
                    this.outputs = [];
                    entries.forEach(([key, params], idx) => {
                        const type = (params.type || "STRING").toUpperCase();
                        const displayName = params.name || key;
                        
                        this.addOutput(key, type);
                        const newOutput = this.outputs[this.outputs.length - 1];
                        newOutput.label = displayName;
                        
                        const backup = oldLinks.find(l => l.name === key);
                        if (backup) {
                            backup.connections.forEach(conn => {
                                this.connect(idx, conn.target_id, conn.target_slot);
                            });
                        }
                    });
                    
                    this.setSize(this.computeSize());
                    this.setDirtyCanvas(true, true);
                };
                
                return r;
            };
            
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, linkInfo) {
                if (onConnectionsChange) onConnectionsChange.apply(this, arguments);
                
                if (type === 1 && slotIndex === 0) {
                    setTimeout(() => this.syncFromUpstream(), 50);
                }
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (onConfigure) onConfigure.apply(this, arguments);
                setTimeout(() => this.syncFromUpstream(), 100);
            };
        }
    }
});