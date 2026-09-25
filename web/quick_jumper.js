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
        // 2. 菜单构建与注入
        // =========================================================================
        function injectJumpItem(menuInstance, target, graph) {
            const rootEl = menuInstance.root;
            if (!rootEl) return;

            // A. 外联端口向内部跳转
            if (target.kind === "subgraph_input") {
                const subIn = target.data;
                const targets = [];
                const linkIds = subIn?.links || (subIn?.link != null ? [subIn.link] : []);

                for (const lid of linkIds) {
                    const link = getLink(graph, lid);
                    if (!link) continue;
                    const destNode = graph.getNodeById(link.target_id);
                    if (!destNode) continue;
                    const slotName = destNode.inputs?.[link.target_slot]?.name || link.target_slot;
                    const title = destNode.title || destNode.type || `#${link.target_id}`;
                    targets.push({
                        label: `[${title}] . ${slotName}`,
                        node: destNode,
                        slotIndex: link.target_slot
                    });
                }

                // 反向扫描兜底
                if (targets.length === 0 && graph._nodes) {
                    for (const n of graph._nodes) {
                        if (!n.inputs) continue;
                        for (let i = 0; i < n.inputs.length; i++) {
                            const inp = n.inputs[i];
                            if (inp.link != null) {
                                const l = getLink(graph, inp.link);
                                if (l && (l.origin_id == null || l.origin_id < 0 || l.origin_id === "inputs" || l.origin_slot === subIn?.id)) {
                                    targets.push({
                                        label: `[${n.title || n.type || n.id}] . ${inp.name || i}`,
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
                    // 【优化】：使用防重复堆叠的专属子菜单控制器
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

            // B. 内部节点跳转
            if (target.kind === "node_slot") {
                const { node, isInput, slotIndex, slotDef } = target;

                if (isInput && slotDef.link != null) {
                    const link = getLink(graph, slotDef.link);
                    if (link) {
                        const originNode = graph.getNodeById(link.origin_id);

                        // 反跳外联端口
                        if (!originNode || link.origin_id < 0 || link.origin_id === "inputs") {
                            const subInName = findSubgraphInputName(graph, slotDef.link) || "外联端口";

                            createDOMMenuItem(rootEl, `Jump to: [${subInName}]`, () => {
                                executeReverseJump(graph, link, node, slotIndex, subInName);
                                menuInstance.close();
                            });
                            return;
                        }

                        // 正常节点跳转
                        const originSlot = originNode.outputs?.[link.origin_slot];
                        const title = originNode.title || originNode.type || `#${link.origin_id}`;
                        const slotName = originSlot?.name || link.origin_slot;

                        createDOMMenuItem(rootEl, `Jump to: [${title}] . ${slotName}`, () => {
                            jumpAndHighlightSlot(originNode, false, link.origin_slot);
                            menuInstance.close();
                        });
                    }
                } else if (!isInput && slotDef.links && slotDef.links.length > 0) {
                    // 输出端多连线处理
                    const targets = [];
                    for (const lid of slotDef.links) {
                        const link = getLink(graph, lid);
                        if (!link) continue;
                        const dest = graph.getNodeById(link.target_id);
                        if (!dest) continue;
                        targets.push({
                            label: `[${dest.title || dest.type}] . ${dest.inputs?.[link.target_slot]?.name || link.target_slot}`,
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
                        // 【优化】：使用防重复堆叠的专属子菜单控制器
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
        // 核心优化：防重复堆叠的子菜单组件 (支持 Toggle 点击切换 + 联动关闭)
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

            itemEl.addEventListener("mouseenter", () => {
                itemEl.style.backgroundColor = "#2a3942";
            });

            itemEl.addEventListener("mouseleave", () => {
                itemEl.style.backgroundColor = "transparent";
            });

            itemEl.addEventListener("click", (e) => {
                e.stopPropagation();

                // Toggle：展开/收起切换
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
                        parentMenu: menuInstance
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

        // =========================================================================
        // 单项普通菜单生成器
        // =========================================================================
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

        function findSubgraphInputName(graph, linkId) {
            if (!graph || linkId == null) return null;

            const link = getLink(graph, linkId);
            if (!link) return null;
            const slotIdx = link.origin_slot;

            // =====================================================================
            // 途径 1：直接从虚拟输入节点 inputNode 提取 (用户 Rename 后这里必定实时更新)
            // =====================================================================
            const inputNode = graph.inputNode || graph._inputNode || (graph.getNodeById && graph.getNodeById(link.origin_id));
            if (inputNode && inputNode.outputs) {
                // A. 按 slotIdx 下标取
                if (slotIdx != null && inputNode.outputs[slotIdx]) {
                    const s = inputNode.outputs[slotIdx];
                    const customName = s.label || s.name;
                    if (customName) return customName;
                }
                // B. 按 link ID 匹配查找该 slot
                const foundSlot = inputNode.outputs.find(s => s.links && s.links.includes(linkId));
                if (foundSlot) {
                    return foundSlot.label || foundSlot.name;
                }
            }

            // =====================================================================
            // 途径 2：从 Subgraph.inputs (或 _inputs) 集合查找
            // =====================================================================
            const inputs = graph.inputs || graph._inputs;
            if (inputs) {
                // 如果是 Map 对象
                if (inputs instanceof Map) {
                    // 先看是否包含以 slotIdx 为 key 的项
                    if (inputs.has(slotIdx)) {
                        const item = inputs.get(slotIdx);
                        return item.label || item.name;
                    }
                    // 遍历 Map 中所有项查找
                    let i = 0;
                    for (const [key, item] of inputs.entries()) {
                        // 匹配下标、id、或包含此 linkId
                        if (i === slotIdx || item.id === slotIdx || item.slot === slotIdx || item.link === linkId || (item.links && item.links.includes(linkId))) {
                            return item.label || item.name || key;
                        }
                        i++;
                    }
                } 
                // 如果是 Array 或 Object
                else {
                    const list = Array.isArray(inputs) ? inputs : Object.values(inputs);
                    // 1. 下标直接命中
                    if (slotIdx != null && list[slotIdx]) {
                        const item = list[slotIdx];
                        return item.label || item.name;
                    }
                    // 2. 属性匹配 (支持 Rename 后的自定义 label / name)
                    for (const item of list) {
                        if (item.id === slotIdx || item.slot === slotIdx || item.link === linkId || (item.links && item.links.includes(linkId))) {
                            return item.label || item.name;
                        }
                    }
                }
            }

            // =====================================================================
            // 途径 3：从外部主图的 GroupNode 映射表查询 (防止子图内没存)
            // =====================================================================
            if (graph.subgraphData?.inputs) {
                const subInputs = graph.subgraphData.inputs;
                if (subInputs[slotIdx]) {
                    return subInputs[slotIdx].label || subInputs[slotIdx].name;
                }
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