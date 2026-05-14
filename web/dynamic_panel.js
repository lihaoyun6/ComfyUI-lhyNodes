import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "lhyNodes.DynamicParameterPanel",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DynamicParameterPanel") {
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                if (onConfigure) onConfigure.apply(this, arguments);
                if (info && info.widgets_values) {
                    const savedJson = info.widgets_values[0];
                    this.buildDynamicUI(savedJson, true); 
                    for (let i = 0; i < info.widgets_values.length; i++) {
                        if (this.widgets[i]) this.widgets[i].value = info.widgets_values[i];
                    }
                    this.refreshLockState(); // 恢复后立即刷新锁定
                }
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.outputs = [];
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

                this.buildDynamicUI = async function(customJsonStr = null, isRestoring = false) {
                    let jsonStr = customJsonStr || this.widgets.find(w => w.name === "config_json")?.value;
                    if (jsonStr === "") jsonStr = "{}";
                    if (!jsonStr) return;

                    let config;
                    try { config = JSON.parse(jsonStr); } catch (e) {
                        if (!isRestoring) alert("Invalid JSON!\n" + e.message); return;
                    }
                    
                    const configKeys = Object.keys(config);
                    if (configKeys.length > 24) {
                        if (!isRestoring) alert(`Too many keys (${configKeys.length}). Max allowed is 24!`);
                        return;
                    }

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

                    for (const [key, params] of Object.entries(config)) {
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

                    // 端口同步逻辑
                    if (!isRestoring) {
                        const oldLinks = (this.outputs || []).map(o => ({ name: o.name, conns: (o.links || []).map(id => {
                            const l = app.graph.links[id]; return l ? { t_id: l.target_id, t_s: l.target_slot } : null;
                        }).filter(x => x) }));
                        this.outputs = [];
                        Object.keys(config).forEach((key, idx) => {
                            const out = this.addOutput(key, (config[key].type || "STRING").toUpperCase());
                            out.label = config[key].name || key;
                            const bk = oldLinks.find(l => l.name === key);
                            if (bk) bk.conns.forEach(c => this.connect(idx, c.t_id, c.t_s));
                        });
                    }

                    this.refreshLockState();
                };

                // 首次创建即刷新锁定状态（隐藏 is_locked）
                setTimeout(() => this.refreshLockState(), 1);

                return r;
            };
        }
    }
});