import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "lhyNodes.QuickJumper",
    setup() {
        if (LGraphNode.prototype.hasOwnProperty("getSlotMenuOptions")) {
            delete LGraphNode.prototype.getSlotMenuOptions;
        }

        const OrigContextMenu = LiteGraph.ContextMenu;

        LiteGraph.ContextMenu = function (values, options) {
            const menuInstance = new OrigContextMenu(values, options);

            // 拦截二级子菜单，防无限套娃
            if (options?.parentMenu || options?.fromSlotJumper) {
                return menuInstance;
            }

            try {
                const canvas = app.canvas;
                const graph = canvas.graph || app.graph;
                const targetSlot = resolveTargetSlot(graph, canvas, options);

                if (targetSlot) {
                    injectJumpItem(menuInstance, targetSlot, graph);
                }
            } catch (err) {
                console.error("[SlotJumper Error]:", err);
            }

            return menuInstance;
        };
        LiteGraph.ContextMenu.prototype = OrigContextMenu.prototype;

        // =========================================================================
        // 1. 槽位探测器
        // =========================================================================
        function resolveTargetSlot(graph, canvas, options) {
            const slotTitle = options?.title;

            // 1. Subgraph 边界外联槽
            if (slotTitle && graph) {
                const subIn = findInSubgraphCollections(graph.inputs || graph._inputs, slotTitle);
                if (subIn) return { kind: "subgraph_input", data: subIn, name: slotTitle };

                const subOut = findInSubgraphCollections(graph.outputs || graph._outputs, slotTitle);
                if (subOut) return { kind: "subgraph_output", data: subOut, name: slotTitle };
            }

            // 2. 普通节点与 Widget 控制槽
            const mousePos = canvas.graph_mouse;
            if (graph._nodes && mousePos) {
                for (const node of graph._nodes) {
                    const slotInfo = node.getSlotInPosition(mousePos[0], mousePos[1]);
                    if (slotInfo) {
                        const isInput = slotInfo.input != null;
                        const slotIndex = slotInfo.slot;
                        const slotDef = isInput ? node.inputs[slotIndex] : node.outputs[slotIndex];
                        return { kind: "node_slot", node, isInput, slotIndex, slotDef };
                    }
                }
            }

            return null;
        }

        function findInSubgraphCollections(coll, name) {
            if (!coll) return null;
            if (coll instanceof Map) {
                for (const [k, v] of coll) {
                    if (v.name === name || v.label === name || k === name) return v;
                }
            } else if (Array.isArray(coll)) {
                return coll.find(v => v.name === name || v.label === name);
            } else if (typeof coll === "object") {
                for (const k in coll) {
                    const v = coll[k];
                    if (v && (v.name === name || v.label === name || k === name)) return v;
                }
            }
            return null;
        }

        // =========================================================================
        // 核心辅助：判断一根线是否【仅属于】当前被右键点击的这个外联输入端口
        // =========================================================================
        function isLinkFromThisSubgraphInput(link, subIn, targetName, graph) {
            if (!link) return false;

            const inputNode = graph.inputNode || graph._inputNode;
            const isOuter = (link.origin_id == null || link.origin_id < 0 || link.origin_id === "inputs" || (inputNode && link.origin_id === inputNode.id));
            if (!isOuter) return false;

            const linkId = link.id;

            if (subIn) {
                if (subIn.link === linkId) return true;
                if (Array.isArray(subIn.links) && subIn.links.includes(linkId)) return true;
                if (subIn.id != null && link.origin_slot === subIn.id) return true;
                if (subIn.slot != null && link.origin_slot === subIn.slot) return true;
            }

            const inputsList = graph.inputs instanceof Map
                ? Array.from(graph.inputs.values())
                : (Array.isArray(graph.inputs) ? graph.inputs : Object.values(graph.inputs || {}));
            
            let subInIndex = subIn ? inputsList.indexOf(subIn) : -1;
            if (subInIndex === -1 && targetName) {
                subInIndex = inputsList.findIndex(item => item.name === targetName || item.label === targetName);
            }
            if (subInIndex !== -1 && link.origin_slot === subInIndex) {
                return true;
            }

            if (inputNode && inputNode.outputs) {
                const slot = inputNode.outputs[link.origin_slot];
                if (slot && (slot.name === targetName || slot.label === targetName)) {
                    return true;
                }
                const targetSlotIdx = inputNode.outputs.findIndex(s => s.name === targetName || s.label === targetName);
                if (targetSlotIdx !== -1) {
                    if (link.origin_slot === targetSlotIdx) return true;
                    if (inputNode.outputs[targetSlotIdx].links && inputNode.outputs[targetSlotIdx].links.includes(linkId)) return true;
                }
            }

            if (link.origin_name && targetName && link.origin_name === targetName) {
                return true;
            }

            return false;
        }

        // =========================================================================
        // 核心辅助：多维获取插槽真实对外展示名称 (优先 label，次选 name)
        // =========================================================================
        function getRealSlotName(destNode, link, isInput, graph) {
            if (!destNode || !link) return "";
            const slotIndex = isInput ? link.target_slot : link.origin_slot;
            const slot = isInput ? destNode.inputs?.[slotIndex] : destNode.outputs?.[slotIndex];

            // 1. 优先读取插槽自身的 label (用户自定义 Rename 的名字)
            if (slot?.label) return slot.label;

            // 2. 如果是连向子图外联端口（无论是子图内部输出端还是主图子图节点），深度解算真实别名
            const customSubName = findSubgraphPortLabel(graph, destNode, link, isInput);
            if (customSubName) return customSubName;

            // 3. 次选原始 name，最后退回索引数字
            return slot?.name || slotIndex;
        }

        // 深度解算子图外联端口的 Rename 名称
        function findSubgraphPortLabel(graph, destNode, link, isInput) {
            if (!destNode || !link) return null;

            // 场景 A：在子图内部连向右侧外联输出代理 (id: -11 / outputNode)
            if (destNode.id === -11 || destNode === graph.outputNode) {
                const outputs = graph.outputs || graph._outputs;
                if (outputs) {
                    const list = outputs instanceof Map ? Array.from(outputs.values()) : (Array.isArray(outputs) ? outputs : Object.values(outputs));
                    const slotIdx = link.target_slot;
                    if (slotIdx != null && list[slotIdx]) {
                        return list[slotIdx].label || list[slotIdx].name;
                    }
                    for (const item of list) {
                        if (item.id === slotIdx || item.slot === slotIdx || item.link === link.id || (item.links && item.links.includes(link.id))) {
                            return item.label || item.name;
                        }
                    }
                }
            }

            // 场景 B：在主图上，上游节点连向了一个 Subgraph 节点
            if (destNode.subgraph) {
                const subInputs = destNode.subgraph.inputs || destNode.subgraph._inputs;
                if (subInputs) {
                    const list = subInputs instanceof Map ? Array.from(subInputs.values()) : (Array.isArray(subInputs) ? subInputs : Object.values(subInputs));
                    if (list[link.target_slot]) {
                        return list[link.target_slot].label || list[link.target_slot].name;
                    }
                }
            }

            return null;
        }

        // =========================================================================
        // 2. 菜单构建与注入
        // =========================================================================
        function injectJumpItem(menuInstance, target, graph) {
            const rootEl = menuInstance.root;
            if (!rootEl) return;

            // =====================================================================
            // A. 外联端口向内部跳转
            // =====================================================================
            if (target.kind === "subgraph_input") {
                const subIn = target.data;
                const targetName = target.name || subIn?.name || subIn?.label;
                const targets = [];

                if (graph._nodes) {
                    for (const n of graph._nodes) {
                        if (n.id === -10 || n.id === -11 || !n.inputs) continue;
                        for (let i = 0; i < n.inputs.length; i++) {
                            const inp = n.inputs[i];
                            if (inp.link != null) {
                                const l = getLink(graph, inp.link);
                                if (l && isLinkFromThisSubgraphInput(l, subIn, targetName, graph)) {
                                    // 优先读取 label
                                    const slotName = inp.label || inp.name || i;
                                    const title = n.title || n.type || `#${n.id}`;
                                    targets.push({
                                        label: `[${title}] . ${slotName}`,
                                        node: n,
                                        slotIndex: i
                                    });
                                }
                            }
                        }
                    }
                }

                if (targets.length === 1) {
                    createDOMMenuItem(rootEl, `Jump to: ${targets[0].label}`, () => {
                        jumpAndHighlightSlot(targets[0].node, true, targets[0].slotIndex);
                        menuInstance.close();
                    });
                } else if (targets.length > 1) {
                    createDOMSubmenuItem(
                        rootEl,
                        menuInstance,
                        `Jump to (${targets.length} slots)...`,
                        targets.map(t => ({
                            label: t.label,
                            callback: () => jumpAndHighlightSlot(t.node, true, t.slotIndex)
                        }))
                    );
                }
                return;
            }

            // =====================================================================
            // B. 内部节点跳转
            // =====================================================================
            if (target.kind === "node_slot") {
                const { node, isInput, slotIndex, slotDef } = target;

                if (isInput && slotDef.link != null) {
                    const link = getLink(graph, slotDef.link);
                    if (link) {
                        const originNode = graph.getNodeById(link.origin_id);

                        // 反跳外联端口
                        if (!originNode || link.origin_id < 0 || link.origin_id === "inputs") {
                            const subInName = findSubgraphInputName(graph, slotDef.link) || "*";

                            createDOMMenuItem(rootEl, `Jump to: [${subInName}]`, () => {
                                executeReverseJump(graph, link, node, slotIndex, subInName);
                                menuInstance.close();
                            });
                            return;
                        }

                        // 正常节点跳转 (优先读取 originSlot.label)
                        const originSlot = originNode.outputs?.[link.origin_slot];
                        const title = originNode.title || originNode.type || `#${link.origin_id}`;
                        const slotName = originSlot?.label || originSlot?.name || link.origin_slot;

                        createDOMMenuItem(rootEl, `Jump to: [${title}] . ${slotName}`, () => {
                            jumpAndHighlightSlot(originNode, false, link.origin_slot);
                            menuInstance.close();
                        });
                    }
                } else if (!isInput && slotDef.links && slotDef.links.length > 0) {
                    // =========================================================
                    // 输出端多连线：修复只能识别成 .source 的核心处
                    // =========================================================
                    const targets = [];
                    for (const lid of slotDef.links) {
                        const link = getLink(graph, lid);
                        if (!link) continue;
                        const dest = graph.getNodeById(link.target_id);
                        if (!dest) continue;

                        // 节点标题 (如果是外联输出虚拟代理则友好命名)
                        const nodeTitle = (dest.id === -11 || dest === graph.outputNode)
                            ? "外联输出"
                            : (dest.title || dest.type || `#${dest.id}`);

                        // 【核心修复】：全面解算真实的 slotName，绝对避开默认的 .source
                        const realSlotName = getRealSlotName(dest, link, true, graph);

                        targets.push({
                            label: `[${nodeTitle}] . ${realSlotName}`,
                            node: dest,
                            slotIndex: link.target_slot
                        });
                    }

                    if (targets.length === 1) {
                        createDOMMenuItem(rootEl, `Jump to: ${targets[0].label}`, () => {
                            jumpAndHighlightSlot(targets[0].node, true, targets[0].slotIndex);
                            menuInstance.close();
                        });
                    } else if (targets.length > 1) {
                        createDOMSubmenuItem(
                            rootEl,
                            menuInstance,
                            `Jump to (${targets.length} slots)...`,
                            targets.map(t => ({
                                label: t.label,
                                callback: () => jumpAndHighlightSlot(t.node, true, t.slotIndex)
                            }))
                        );
                    }
                }
            }
        }

        // =========================================================================
        // 核心辅助：准确反查外联端点的真实自定义 Rename 名称
        // =========================================================================
        function findSubgraphInputName(graph, linkId) {
            if (!graph || linkId == null) return null;

            const link = getLink(graph, linkId);
            if (!link) return null;
            const slotIdx = link.origin_slot;

            const inputNode = graph.inputNode || graph._inputNode || (graph.getNodeById && graph.getNodeById(link.origin_id));
            if (inputNode && inputNode.outputs) {
                if (slotIdx != null && inputNode.outputs[slotIdx]) {
                    const s = inputNode.outputs[slotIdx];
                    if (s.label || s.name) return s.label || s.name;
                }
                const foundSlot = inputNode.outputs.find(s => s.links && s.links.includes(linkId));
                if (foundSlot) return foundSlot.label || foundSlot.name;
            }

            const inputs = graph.inputs || graph._inputs;
            if (inputs) {
                const list = inputs instanceof Map ? Array.from(inputs.values()) : (Array.isArray(inputs) ? inputs : Object.values(inputs));
                if (slotIdx != null && list[slotIdx]) {
                    const item = list[slotIdx];
                    return item.label || item.name;
                }
                for (const item of list) {
                    if (item.id === slotIdx || item.slot === slotIdx || item.link === linkId || (item.links && item.links.includes(linkId))) {
                        return item.label || item.name;
                    }
                }
            }

            return null;
        }

        // =========================================================================
        // 界面组件：对齐原生边距 + 自动单箭头
        // =========================================================================
        function createDOMSubmenuItem(menuRoot, menuInstance, text, optionsList) {
            const itemEl = document.createElement("div");
            itemEl.className = "litemenu-entry has_submenu";
            itemEl.style.cssText = "color: #00e6ff; font-weight: bold; border-bottom: 1px solid #444; padding: 4px 2px; padding-right: 12px; cursor: pointer;";
            itemEl.innerText = text;

            let activeSubmenu = null;

            const closeSubmenu = () => {
                if (activeSubmenu) {
                    activeSubmenu.close();
                    activeSubmenu = null;
                }
                if (menuInstance.current_submenu) {
                    menuInstance.current_submenu = null;
                }
            };

            itemEl.addEventListener("mouseenter", () => itemEl.style.backgroundColor = "#2a3942");
            itemEl.addEventListener("mouseleave", () => itemEl.style.backgroundColor = "transparent");

            itemEl.addEventListener("click", (e) => {
                e.stopPropagation();

                if (activeSubmenu) {
                    closeSubmenu();
                    return;
                }

                if (menuInstance.current_submenu) {
                    menuInstance.current_submenu.close();
                    menuInstance.current_submenu = null;
                }

                activeSubmenu = new LiteGraph.ContextMenu(
                    optionsList.map(opt => ({
                        content: opt.label,
                        callback: () => {
                            if (typeof opt.callback === "function") {
                                opt.callback();
                            }
                            closeSubmenu();
                            menuInstance.close();
                        }
                    })),
                    {
                        event: e,
                        parentMenu: menuInstance,
                        fromSlotJumper: true
                    }
                );

                menuInstance.current_submenu = activeSubmenu;

                const origClose = activeSubmenu.close;
                activeSubmenu.close = function () {
                    activeSubmenu = null;
                    if (menuInstance.current_submenu === this) {
                        menuInstance.current_submenu = null;
                    }
                    return origClose ? origClose.apply(this, arguments) : undefined;
                };
            });

            menuRoot.insertBefore(itemEl, menuRoot.firstChild);
        }

        function createDOMMenuItem(menuRoot, text, onClick) {
            const itemEl = document.createElement("div");
            itemEl.className = "litemenu-entry";
            itemEl.style.cssText = "color: #00e6ff; font-weight: bold; border-bottom: 1px solid #444; padding: 4px 2px; cursor: pointer;";
            itemEl.innerText = text;

            itemEl.addEventListener("mouseenter", () => itemEl.style.backgroundColor = "#2a3942");
            itemEl.addEventListener("mouseleave", () => itemEl.style.backgroundColor = "transparent");
            itemEl.addEventListener("click", (e) => {
                e.stopPropagation();
                onClick(e);
            });

            menuRoot.insertBefore(itemEl, menuRoot.firstChild);
        }

        // =========================================================================
        // 探针反向跳转 (中点反推绝对起点)
        // =========================================================================
        function executeReverseJump(graph, link, innerNode, innerSlotIdx, subInName) {
            let finalPos = null;

            const inputs = graph.inputs || graph._inputs;
            const list = inputs instanceof Map ? Array.from(inputs.values()) : (Array.isArray(inputs) ? inputs : Object.values(inputs || {}));
            const subItem = list.find(i => (i.name || i.label) === subInName);

            if (subItem && subItem.pos && !isNaN(subItem.pos[0])) {
                finalPos = subItem.pos;
            }

            if (!finalPos && link) {
                if (link.origin_pos && !isNaN(link.origin_pos[0])) {
                    finalPos = link.origin_pos;
                } else if (link.points && link.points.length >= 2 && !isNaN(link.points[0])) {
                    finalPos = [link.points[0], link.points[1]];
                }
            }

            // 中点逆解
            if (!finalPos && link && link._pos && !isNaN(link._pos[0])) {
                const endPos = innerNode.getConnectionPos(true, innerSlotIdx);
                const startX = 2 * link._pos[0] - endPos[0];
                const startY = 2 * link._pos[1] - endPos[1];
                finalPos = [startX, startY];
            }

            if (finalPos) {
                jumpToPos(finalPos);
            } else {
                const p = innerNode.getConnectionPos(true, innerSlotIdx);
                jumpToPos([p[0] - 300, p[1]]);
            }
        }

        // =========================================================================
        // 辅助与定位系统
        // =========================================================================
        function getLink(graph, linkId) {
            if (linkId == null || !graph) return null;
            const pool = graph.links || graph._links;
            if (pool instanceof Map) {
                return pool.get(linkId) || pool.get(Number(linkId)) || pool.get(String(linkId));
            }
            if (typeof pool === "object") {
                return pool[linkId];
            }
            return null;
        }

        let highlightData = null;
        let animStartTime = 0;

        function jumpToPos(pos) {
            if (!pos || isNaN(pos[0]) || isNaN(pos[1])) return;
            const canvasEl = app.canvas.canvas;
            const rect = canvasEl.getBoundingClientRect();
            const scale = app.canvas.ds.scale;

            app.canvas.ds.offset[0] = (rect.width / (2 * scale)) - pos[0];
            app.canvas.ds.offset[1] = (rect.height / (2 * scale)) - pos[1];

            highlightData = { directPos: pos };
            animStartTime = performance.now();
            app.canvas.setDirty(true, true);
        }

        function jumpAndHighlightSlot(targetNode, isInput, slotIndex) {
            if (!targetNode) return;
            const pos = targetNode.getConnectionPos(isInput, slotIndex);
            jumpToPos(pos);
            highlightData = { targetNode, isInput, slotIndex };
        }

        // 高亮波纹
        const origDrawForeground = app.canvas.onDrawForeground;
        app.canvas.onDrawForeground = function (ctx) {
            origDrawForeground?.apply(this, arguments);

            if (highlightData) {
                const elapsed = performance.now() - animStartTime;
                const duration = 1200;

                if (elapsed > duration) {
                    highlightData = null;
                } else {
                    const progress = elapsed / duration;
                    let pos;
                    if (highlightData.directPos) {
                        pos = highlightData.directPos;
                    } else if (highlightData.targetNode) {
                        pos = highlightData.targetNode.getConnectionPos(
                            highlightData.isInput,
                            highlightData.slotIndex
                        );
                    }

                    if (pos) {
                        ctx.save();
                        ctx.beginPath();
                        const radius = 6 + progress * 32;
                        ctx.arc(pos[0], pos[1], radius, 0, Math.PI * 2);
                        ctx.strokeStyle = `rgba(0, 230, 255, ${1 - progress})`;
                        ctx.lineWidth = 3;
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.arc(pos[0], pos[1], 7, 0, Math.PI * 2);
                        ctx.fillStyle = "#00e6ff";
                        ctx.shadowColor = "#00e6ff";
                        ctx.shadowBlur = 15;
                        ctx.fill();
                        ctx.restore();

                        app.canvas.setDirty(true, false);
                    }
                }
            }
        };
    }
});