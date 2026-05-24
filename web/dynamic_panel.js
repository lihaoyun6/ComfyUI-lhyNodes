import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const findAllNodes = (nodes, type) => {
    let found = [];
    for (const node of nodes) {
        if (!type || node.type === type) found.push(node);
        if (node.subgraph && node.subgraph._nodes) {
            found.push(...findAllNodes(node.subgraph._nodes, type));
        } else if (typeof node.getInnerNodes === 'function') {
            try {
                const inner = node.getInnerNodes();
                if (inner) found.push(...findAllNodes(inner, type));
            } catch (e) {}
        }
    }
    return found;
};

app.registerExtension({
    name: "lhyNodes.DynamicParameterPanel",
    
    setup() {
        const originalGraphToPrompt = app.graphToPrompt;
        
        app.graphToPrompt = async function () {
            const res = await originalGraphToPrompt.apply(this, arguments);
            const prompt = res.output; 
            
            try {
                const panelOutputs = {};
                const replaceMap = {};
                
                for (const [nodeId, node] of Object.entries(prompt)) {
                    if (node.class_type === "DynamicParameterPanel") {
                        const configJson = node.inputs?.config_json || "{}";
                        let config = {};
                        try { config = JSON.parse(configJson); } catch(e) {}
                        
                        const entries = Object.entries(config).slice(0, 32).filter(([k, p]) => {
                            const t = (p.type || "STRING").toUpperCase();
                            return t !== "BUTTON" && t !== "MUTER";
                        });
                        
                        panelOutputs[nodeId] = {};
                        entries.forEach(([key, params], slotIdx) => {
                            const type = (params.type || "STRING").toUpperCase();
                            if (type === "INPUT") {
                                const rawOpt = params.optional || false;
                                const displayKey = rawOpt ? `${key} (opt)` : key;
                                const inLink = node.inputs[displayKey];
                                
                                if (inLink && Array.isArray(inLink)) {
                                    panelOutputs[nodeId][slotIdx] = inLink;
                                } else {
                                    if (rawOpt) {
                                        panelOutputs[nodeId][slotIdx] = null;
                                    } else {
                                        alert(`Validation Error: Required input '${key}' is missing!`);
                                        throw new Error(`Required input missing`);
                                    }
                                }
                            } else {
                                let val = node.inputs[key];
                                if (val === undefined) val = params.default;
                                if (type === "INT" || type === "SEED") val = Math.round(Number(val));
                                else if (type === "FLOAT") val = Number(val);
                                else if (type === "BOOLEAN") val = Boolean(val);
                                else if (type === "STRING") val = String(val);
                                panelOutputs[nodeId][slotIdx] = val;
                            }
                        });
                    }
                }
                            
                for (const [nodeId, node] of Object.entries(prompt)) {
                    if (node.class_type === "ParameterUnpacker") {
                        let panelId = null;
                        if (node.inputs) {
                            for (const val of Object.values(node.inputs)) {
                                if (Array.isArray(val) && val.length === 2) {
                                    panelId = String(val[0]);
                                    break;
                                }
                            }
                        }
                        if (panelId && panelOutputs[panelId]) {
                            replaceMap[nodeId] = panelOutputs[panelId];
                        }
                    }
                }
                            
                for (const [nodeId, node] of Object.entries(prompt)) {
                    if (node.class_type === "DynamicParameterPanel" || node.class_type === "ParameterUnpacker") continue;
                    
                    if (node.inputs) {
                        for (const [inKey, inVal] of Object.entries(node.inputs)) {
                            if (Array.isArray(inVal) && inVal.length === 2) {
                                const sourceId = String(inVal[0]);
                                const sourceSlot = inVal[1];
                                
                                if (replaceMap[sourceId] && replaceMap[sourceId][sourceSlot] !== undefined) {
                                    const replaceVal = replaceMap[sourceId][sourceSlot];
                                    if (replaceVal === null) {
                                        delete node.inputs[inKey];
                                    } else {
                                        node.inputs[inKey] = replaceVal;
                                    }
                                }
                            }
                        }
                    }
                }
                    
                for (const nodeId of Object.keys(prompt)) {
                    if (prompt[nodeId].class_type === "DynamicParameterPanel" || prompt[nodeId].class_type === "ParameterUnpacker") {
                        delete prompt[nodeId];
                    }
                }
            } catch(e) {
                console.error("[DynamicPanel] Graph interception failed:", e);
            }
            return res; 
        };
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DynamicParameterPanel") {
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                this.isConfigured = true; 
                if (onConfigure) onConfigure.apply(this, arguments);
                
                if (info && info.widgets_values) {
                    const savedJson = info.widgets_values[0];
                    this.buildDynamicUI(savedJson, true, info.widgets_values); 
                }
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
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

                this.buildDynamicUI = async function(customJsonStr = null, isRestoring = false, storedValues = null) {
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

                    for (let i = this.widgets.length - 1; i >= 0; i--) {
                        if (!isStatic(this.widgets[i])) {
                            if (this.widgets[i].onRemove) this.widgets[i].onRemove();
                            this.widgets.splice(i, 1);
                        }
                    }
                    
                    if (this.inputs) while(this.inputs.length > 0) this.removeInput(this.inputs.length - 1);

                    const oldInputs = [];
                    if (this.inputs) {
                        for (let i = 0; i < this.inputs.length; i++) {
                            const inp = this.inputs[i];
                            if (inp.link) {
                                const l = app.graph.links[inp.link];
                                if (l) {
                                    oldInputs.push({ name: inp.name, origin_id: l.origin_id, origin_slot: l.origin_slot });
                                }
                            }
                        }
                        while(this.inputs.length > 0) {
                            this.removeInput(this.inputs.length - 1);
                        }
                    }

                    for (const [key, params] of entries) {
                        const type = (params.type || "STRING").toUpperCase();
                        let widget;
                        let val = params.default;

                        if (type === "INT" || type === "FLOAT") {
                            const isInt = type === "INT";
                            const step = params.step ?? (isInt ? 1 : 0.1);
                            const prec = params.precision ?? (isInt ? 0 : 3);
                            const isSlider = params.slider === true || params.display === "slider";
                            
                            if (isSlider && (params.min === undefined || params.max === undefined)) {
                                const errMsg = `JSON Config Error:\nSlider '${key}' is missing the required 'min' or 'max' parameters!`;
                                if (!isRestoring) alert(errMsg);
                                console.error("[DynamicPanel]", errMsg);
                                return;
                            }
                            
                            const minVal = params.min ?? -999999;
                            const maxVal = params.max ?? 999999;
                            const initVal = val ?? 0;
                            
                            widget = this.addWidget(isSlider ? "slider" : "number", key, initVal, (v) => {
                                let snapped = isInt ? Math.round(v) : Math.round(v / step) * step;
                                const finalVal = parseFloat(snapped.toFixed(prec));
                                if (widget.value !== finalVal) {
                                    widget.value = finalVal;
                                }
                            }, { 
                                min: minVal, 
                                max: maxVal, 
                                step: isSlider ? step : step * 10, 
                                precision: prec 
                            });
                        } else if (type === "SEED") {
                            const minVal = params.min ?? 0;
                            const maxVal = params.max ?? 0xffffffffffffffff;
                            const initVal = val ?? params.default ?? 0;
                                
                            ComfyWidgets.INT(this, key, ["INT", { 
                                default: initVal, 
                                    min: minVal, 
                                    max: maxVal, 
                                    control_after_generate: true 
                            }], app);
                            
                            widget = this.widgets.find(w => w.name === key);
                            if (widget) widget.value = initVal;
                        } else if (type === "STRING") {
                            if (params.multiline || params.placeholder) {
                                ComfyWidgets.STRING(this, key, ["STRING", { multiline: !!params.multiline, default: val || "" }], app);
                                widget = this.widgets[this.widgets.length - 1];
                                widget.value = val || "";
                                if (widget.inputEl && params.placeholder) widget.inputEl.placeholder = params.placeholder;
                            } else {
                                widget = this.addWidget("text", key, val || "", (v) => {}, {});
                            }
                        } else if (type === "COMBO") {
                            let values = params.values || ["None"];
                            if (params.folder) {
                                try {
                                    const response = await fetch(`/dynamic_panel/files/${params.folder}`);
                                    if (response.ok) values = await response.json();
                                } catch (e) { console.error("Fetch models failed:", e); }
                            }
                            widget = this.addWidget("combo", key, val || values[0], (v) => {}, { values: values });
                        } else if (type === "BOOLEAN") {
                            widget = this.addWidget("toggle", key, val ?? true, (v) => {}, {});
                        } else if (type === "BUTTON") {
                            const action = params.action === "stop" ? "stop" : "run";
                            widget = this.addWidget("button", params.name || key, action, () => {
                                if (action === "run") app.queuePrompt(0);
                                else if (action === "stop") api.interrupt();
                            });
                        } else if (type === "BYPASSER") {
                            const applyMuter = (v) => {
                                const nodes = findAllNodes(app.graph._nodes, null); 
                                const tIds = params.match_id ? (Array.isArray(params.match_id) ? params.match_id : [params.match_id]).map(String) : null;
                                const tTitles = params.match_title ? (Array.isArray(params.match_title) ? params.match_title : [params.match_title]).map(String) : null;
                                nodes.forEach(n => {
                                    let match = false;
                                    const nid = String(n.id);
                                    const ntitle = String(n.title || n.type);
                                    if (tIds && tTitles) match = tIds.includes(nid) && tTitles.includes(ntitle);
                                    else if (tIds) match = tIds.includes(nid);
                                    else if (tTitles) match = tTitles.includes(ntitle);
                                    if (match) n.mode = v ? 0 : 4;
                                });
                            };
                            widget = this.addWidget("toggle", key, val, (v) => applyMuter(v), {});
                            setTimeout(() => applyMuter(widget.value), 300);
                        } else if (type === "INPUT") {
                            const inputClass = params.class ? String(params.class).toUpperCase() : "*";
                            const isOptional = params.optional ?? false;
                            this.addInput(key, inputClass, {shape: isOptional ? 7 : undefined});
                            
                            const newIdx = this.inputs.length - 1;
                            const backup = oldInputs.find(l => l.name === key);
                            if (backup) {
                                const originNode = app.graph.getNodeById(backup.origin_id);
                                if (originNode) originNode.connect(backup.origin_slot, this, newIdx);
                            }
                        }

                        if (widget) {
                            widget.label = params.name || key;
                            widget.tooltip = params.tooltip;
                            
                            if (storedValues) {
                                const wIdx = this.widgets.indexOf(widget);
                                if (wIdx !== -1 && storedValues[wIdx] !== undefined) widget.value = storedValues[wIdx];
                            }
                        }
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
                    
                    const outputEntries = entries.filter(([k, p]) => {
                        const t = (p.type || "STRING").toUpperCase();
                        return t !== "BUTTON" && t !== "BYPASSER";
                    });
                    
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
                    outputEntries.forEach(([key, params], idx) => {
                        const baseType = (params.type || "*").toUpperCase();
                        let outputClass;
                        
                        if (params.class) {
                            outputClass = String(params.class).toUpperCase()
                        } else if (baseType === "INPUT") {
                            outputClass = "*"; 
                        } else if (baseType === "SEED") {
                            outputClass = "INT"; 
                        } else {
                            outputClass = baseType; 
                        }
                        
                        const displayName = params.name || key;
                        const isOptional = params.optional ?? false;
                        
                        this.addOutput(key, outputClass, {shape: isOptional ? 7 : undefined});
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