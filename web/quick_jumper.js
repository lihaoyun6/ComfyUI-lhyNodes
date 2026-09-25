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
        // 1. 槽位探测器 (核心修复：识别 options.extra 就是 ComfyNode)
        // =========================================================================
        function resolveTargetSlot(graph, canvas, options) {
            // 【破案核心 A】：识别 options.extra 本身就是 ComfyNode 实体节点！
            let node = options?.node;
            let slotInfo = options?.extra;

            if (!node && slotInfo && (slotInfo.inputs || slotInfo.outputs || slotInfo.id != null)) {
                node = slotInfo; // extra 实质上就是实体节点！
                slotInfo = null;
            }

            // 如果拿到了普通实体节点 (排除虚拟节点 -10 / -11)
            if (node && node.id !== -10 && node.id !== -11 && node !== graph.inputNode && node !== graph.outputNode) {
                const mousePos = canvas.graph_mouse;
                // 1. 尝试通过坐标获取准确槽位
                if (!slotInfo && mousePos) {
                    slotInfo = node.getSlotInPosition(mousePos[0], mousePos[1]);
                }

                if (slotInfo) {
                    const isInput = slotInfo.input != null;
                    const slotIndex = slotInfo.slot !== undefined ? slotInfo.slot : (isInput ? node.inputs?.indexOf(slotInfo.input) : node.outputs?.indexOf(slotInfo.output));
                    if (slotIndex !== -1 && slotIndex !== undefined) {
                        const slotDef = isInput ? node.inputs?.[slotIndex] : node.outputs?.[slotIndex];
                        return { kind: "node_slot", node, isInput, slotIndex, slotDef };
                    }
                }

                // 2. 如果 slotInfo 未取到，通过 options.title 在 node 的 inputs/outputs 匹配
                const title = options?.title;
                if (title) {
                    // 优先检查输出槽
                    const outIdx = node.outputs?.findIndex(s => s.name === title || s.label === title);
                    if (outIdx !== -1 && outIdx !== undefined) {
                        return { kind: "node_slot", node, isInput: false, slotIndex: outIdx, slotDef: node.outputs[outIdx] };
                    }
                    // 其次检查输入槽
                    const inIdx = node.inputs?.findIndex(s => s.name === title || s.label === title);
                    if (inIdx !== -1 && inIdx !== undefined) {
                        return { kind: "node_slot", node, isInput: true, slotIndex: inIdx, slotDef: node.inputs[inIdx] };
                    }
                }
            }

            // 【破案核心 B】：虚拟外联节点 (inputNode: -10, outputNode: -11)
            if (node && (node.id === -10 || node.id === -11 || node === graph.inputNode || node === graph.outputNode)) {
                const isOut = node.id === -11 || node === graph.outputNode;
                const slotIndex = slotInfo?.slot !== undefined ? slotInfo.slot : 0;
                const slotDef = isOut ? node.inputs?.[slotIndex] : node.outputs?.[slotIndex];
                const name = options?.title || slotDef?.label || slotDef?.name;
                return {
                    kind: isOut ? "subgraph_output" : "subgraph_input",
                    node: node,
                    slotIndex: slotIndex,
                    slotDef: slotDef,
                    data: isOut ? findInSubgraphCollections(graph.outputs || graph._outputs, name) : findInSubgraphCollections(graph.inputs || graph._inputs, name),
                    name: name
                };
            }

            // 【破案核心 C】：无实体 node 时的 Subgraph 纯边界外联槽
            const slotTitle = options?.title;
            if (slotTitle && graph && !node) {
                const subIn = findInSubgraphCollections(graph.inputs || graph._inputs, slotTitle);
                if (subIn) return { kind: "subgraph_input", data: subIn, name: slotTitle };

                const subOut = findInSubgraphCollections(graph.outputs || graph._outputs, slotTitle);
                if (subOut) return { kind: "subgraph_output", data: subOut, name: slotTitle };
            }

            // 【破案核心 D】：全图坐标容差探测兜底
            const mousePos = canvas.graph_mouse;
            if (graph._nodes && mousePos) {
                for (const n of graph._nodes) {
                    if (n.id === -10 || n.id === -11) continue;

                    const sInfo = n.getSlotInPosition(mousePos[0], mousePos[1]);
                    if (sInfo) {
                        const isInput = sInfo.input != null;
                        const slotIndex = sInfo.slot;
                        const slotDef = isInput ? n.inputs?.[slotIndex] : n.outputs?.[slotIndex];
                        return { kind: "node_slot", node: n, isInput, slotIndex, slotDef };
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
        // 核心辅助：判断一根线是否仅属于当前【外联输入】
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
            if (subInIndex !== -1 && link.origin_slot === subInIndex) return true;

            if (inputNode && inputNode.outputs) {
                const slot = inputNode.outputs[link.origin_slot];
                if (slot && (slot.name === targetName || slot.label === targetName)) return true;
                const targetSlotIdx = inputNode.outputs.findIndex(s => s.name === targetName || s.label === targetName);
                if (targetSlotIdx !== -1) {
                    if (link.origin_slot === targetSlotIdx) return true;
                    if (inputNode.outputs[targetSlotIdx].links && inputNode.outputs[targetSlotIdx].links.includes(linkId)) return true;
                }
            }

            if (link.origin_name && targetName && link.origin_name === targetName) return true;

            return false;
        }

        // =========================================================================
        // 核心辅助：判断一根线是否仅属于当前【外联输出】
        // =========================================================================
        function isLinkToThisSubgraphOutput(link, subOut, targetName, graph) {
            if (!link) return false;

            const outputNode = graph.outputNode || graph._outputNode;
            const isOuter = (link.target_id == null || link.target_id < 0 || link.target_id === "outputs" || link.target_id === -11 || (outputNode && link.target_id === outputNode.id));
            if (!isOuter) return false;

            const linkId = link.id;

            if (subOut) {
                if (subOut.link === linkId) return true;
                if (Array.isArray(subOut.links) && subOut.links.includes(linkId)) return true;
                if (subOut.id != null && link.target_slot === subOut.id) return true;
                if (subOut.slot != null && link.target_slot === subOut.slot) return true;
            }

            const outputsList = graph.outputs instanceof Map
                ? Array.from(graph.outputs.values())
                : (Array.isArray(graph.outputs) ? graph.outputs : Object.values(graph.outputs || {}));
            
            let subOutIndex = subOut ? outputsList.indexOf(subOut) : -1;
            if (subOutIndex === -1 && targetName) {
                subOutIndex = outputsList.findIndex(item => item.name === targetName || item.label === targetName);
            }
            if (subOutIndex !== -1 && link.target_slot === subOutIndex) return true;

            if (outputNode && outputNode.inputs) {
                const slot = outputNode.inputs[link.target_slot];
                if (slot && (slot.name === targetName || slot.label === targetName)) return true;
                const targetSlotIdx = outputNode.inputs.findIndex(s => s.name === targetName || s.label === targetName);
                if (targetSlotIdx !== -1) {
                    if (link.target_slot === targetSlotIdx) return true;
                    if (outputNode.inputs[targetSlotIdx].link === linkId) return true;
                }
            }

            if (link.target_name && targetName && link.target_name === targetName) return true;

            return false;
        }

        // =========================================================================
        // 核心辅助：获取真实插槽展示名称
        // =========================================================================
        function getRealSlotName(destNode, link, isInput, graph) {
            if (!destNode || !link) return "";
            const slotIndex = isInput ? link.target_slot : link.origin_slot;
            const slot = isInput ? destNode.inputs?.[slotIndex] : destNode.outputs?.[slotIndex];

            if (slot?.label) return slot.label;

            const customSubName = findSubgraphPortLabel(graph, destNode, link, isInput);
            if (customSubName) return customSubName;

            return slot?.name || slotIndex;
        }

        function findSubgraphPortLabel(graph, destNode, link, isInput) {
            if (!destNode || !link) return null;

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
            // A. 外联【输入】端口向内部跳转
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
            // B. 外联【输出】端口向内部源头跳转
            // =====================================================================
            if (target.kind === "subgraph_output") {
                const subOut = target.data;
                const targetName = target.name || subOut?.name || subOut?.label;
                const outputNode = target.node || getSubgraphVirtualNode(graph, false);
                let originTarget = null;

                if (outputNode && outputNode.inputs) {
                    const inSlot = target.slotIndex !== undefined 
                        ? outputNode.inputs[target.slotIndex] 
                        : outputNode.inputs.find(s => s.name === targetName || s.label === targetName);
                    
                    if (inSlot?.link != null) {
                        const l = getLink(graph, inSlot.link);
                        if (l) {
                            const originNode = graph.getNodeById(l.origin_id);
                            if (originNode) {
                                const originSlot = originNode.outputs?.[l.origin_slot];
                                const slotName = originSlot?.label || originSlot?.name || l.origin_slot;
                                const title = originNode.title || originNode.type || `#${originNode.id}`;
                                originTarget = {
                                    label: `[${title}] . ${slotName}`,
                                    node: originNode,
                                    slotIndex: l.origin_slot
                                };
                            }
                        }
                    }
                }

                if (!originTarget && subOut && subOut.link != null) {
                    const l = getLink(graph, subOut.link);
                    if (l) {
                        const originNode = graph.getNodeById(l.origin_id);
                        if (originNode) {
                            const originSlot = originNode.outputs?.[l.origin_slot];
                            const slotName = originSlot?.label || originSlot?.name || l.origin_slot;
                            const title = originNode.title || originNode.type || `#${originNode.id}`;
                            originTarget = {
                                label: `[${title}] . ${slotName}`,
                                node: originNode,
                                slotIndex: l.origin_slot
                            };
                        }
                    }
                }

                if (!originTarget && graph._nodes) {
                    for (const n of graph._nodes) {
                        if (n.id === -10 || n.id === -11 || !n.outputs) continue;
                        for (let oIdx = 0; oIdx < n.outputs.length; oIdx++) {
                            const outSlot = n.outputs[oIdx];
                            if (outSlot.links) {
                                for (const lid of outSlot.links) {
                                    const l = getLink(graph, lid);
                                    if (l && isLinkToThisSubgraphOutput(l, subOut, targetName, graph)) {
                                        const slotName = outSlot.label || outSlot.name || oIdx;
                                        const title = n.title || n.type || `#${n.id}`;
                                        originTarget = {
                                            label: `[${title}] . ${slotName}`,
                                            node: n,
                                            slotIndex: oIdx
                                        };
                                        break;
                                    }
                                }
                            }
                            if (originTarget) break;
                        }
                        if (originTarget) break;
                    }
                }

                if (originTarget) {
                    createDOMMenuItem(rootEl, `Jump to: ${originTarget.label}`, () => {
                        jumpAndHighlightSlot(originTarget.node, false, originTarget.slotIndex);
                        menuInstance.close();
                    });
                }
                return;
            }

            // =====================================================================
            // C. 内部实体节点跳转 (严格保证以实体节点自身展开分析)
            // =====================================================================
            if (target.kind === "node_slot") {
                const { node, isInput, slotIndex, slotDef } = target;

                if (isInput) {
                    const linkId = slotDef?.link;
                    if (linkId != null) {
                        const link = getLink(graph, linkId);
                        if (link) {
                            const originNode = graph.getNodeById(link.origin_id);

                            // 反跳外联输入端口
                            if (!originNode || link.origin_id < 0 || link.origin_id === "inputs") {
                                const subInName = findSubgraphInputName(graph, linkId) || "*";

                                createDOMMenuItem(rootEl, `Jump to: [${subInName}]`, () => {
                                    executeReverseJump(graph, link, node, slotIndex, subInName);
                                    menuInstance.close();
                                });
                                return;
                            }

                            // 正常上游节点跳转
                            const originSlot = originNode.outputs?.[link.origin_slot];
                            const title = originNode.title || originNode.type || `#${link.origin_id}`;
                            const slotName = originSlot?.label || originSlot?.name || link.origin_slot;

                            createDOMMenuItem(rootEl, `Jump to: [${title}] . ${slotName}`, () => {
                                jumpAndHighlightSlot(originNode, false, link.origin_slot);
                                menuInstance.close();
                            });
                        }
                    }
                } else {
                    // 输出端连线分析
                    let linkList = Array.isArray(slotDef?.links) ? [...slotDef.links] : [];

                    if (linkList.length === 0 && graph.links) {
                        const pool = graph.links instanceof Map ? graph.links.values() : Object.values(graph.links);
                        for (const l of pool) {
                            if (l && l.origin_id === node.id && l.origin_slot === slotIndex) {
                                linkList.push(l.id);
                            }
                        }
                    }

                    if (linkList.length > 0) {
                        const targets = [];
                        for (const lid of linkList) {
                            const link = getLink(graph, lid);
                            if (!link) continue;
                            
                            let dest = graph.getNodeById(link.target_id);

                            // 连向外联输出的判定
                            const isSelfLoop = (dest === node || link.target_id === node.id);
                            const isOuterOutput = isSelfLoop || (link.target_id == null || link.target_id < 0 || link.target_id === "outputs" || link.target_id === -11 || dest === graph.outputNode);

                            if (isOuterOutput || !dest) {
                                const outName = findSubgraphOutputName(graph, lid) || "*";
                                targets.push({
                                    label: `[${outName}]`,
                                    isOuterOutput: true,
                                    link: link,
                                    name: outName
                                });
                                continue;
                            }

                            const nodeTitle = dest.title || dest.type || `#${dest.id}`;
                            const realSlotName = getRealSlotName(dest, link, true, graph);

                            targets.push({
                                label: `[${nodeTitle}] . ${realSlotName}`,
                                node: dest,
                                slotIndex: link.target_slot
                            });
                        }

                        if (targets.length === 1) {
                            const t = targets[0];
                            createDOMMenuItem(rootEl, `Jump to: ${t.label}`, () => {
                                if (t.isOuterOutput) {
                                    executeJumpToOuterOutput(graph, t.link, node, slotIndex, t.name);
                                } else {
                                    jumpAndHighlightSlot(t.node, true, t.slotIndex);
                                }
                                menuInstance.close();
                            });
                        } else if (targets.length > 1) {
                            createDOMSubmenuItem(
                                rootEl,
                                menuInstance,
                                `Jump to (${targets.length} slots)...`,
                                targets.map(t => ({
                                    label: t.label,
                                    callback: () => {
                                        if (t.isOuterOutput) {
                                            executeJumpToOuterOutput(graph, t.link, node, slotIndex, t.name);
                                        } else {
                                            jumpAndHighlightSlot(t.node, true, t.slotIndex);
                                        }
                                    }
                                }))
                            );
                        }
                    }
                }
            }
        }

        // =========================================================================
        // 核心辅助：反查外联输出端口的真实自定义名称
        // =========================================================================
        function findSubgraphOutputName(graph, linkId) {
            if (!graph || linkId == null) return null;

            const link = getLink(graph, linkId);
            if (!link) return null;
            const slotIdx = link.target_slot;

            const outputNode = graph.outputNode || graph._outputNode || (graph.getNodeById && graph.getNodeById(link.target_id));
            if (outputNode && outputNode.inputs) {
                if (slotIdx != null && outputNode.inputs[slotIdx]) {
                    const s = outputNode.inputs[slotIdx];
                    if (s.label || s.name) return s.label || s.name;
                }
                const foundSlot = outputNode.inputs.find(s => s.link === linkId || (s.links && s.links.includes(linkId)));
                if (foundSlot) return foundSlot.label || foundSlot.name;
            }

            const outputs = graph.outputs || graph._outputs;
            if (outputs) {
                const list = outputs instanceof Map ? Array.from(outputs.values()) : (Array.isArray(outputs) ? outputs : Object.values(outputs));
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

        function getSubgraphVirtualNode(graph, isInput) {
            if (!graph) return null;
            if (isInput && graph.inputNode) return graph.inputNode;
            if (!isInput && graph.outputNode) return graph.outputNode;

            const targetId = isInput ? -10 : -11;
            if (graph.getNodeById) {
                const n = graph.getNodeById(targetId);
                if (n) return n;
            }
            if (graph._nodes) {
                for (const n of graph._nodes) {
                    if (n.id === targetId) return n;
                }
            }
            return null;
        }

        // =========================================================================
        // 界面组件
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
        // 跳转到外联输出端口 (安全防崩 + 中点正向推导终点)
        // =========================================================================
        function executeJumpToOuterOutput(graph, link, innerNode, innerSlotIdx, outName) {
            let finalPos = null;
            
            // 1. 安全调用 outputNode (用 try-catch 拦截原生 reading 'collapsed' 报错)
            try {
                const outputNode = getSubgraphVirtualNode(graph, false);
                if (outputNode && link) {
                    const slotIdx = link.target_slot ?? 0;
                    // 仅当输入槽真实存在时才尝试调用
                    if (outputNode.inputs && outputNode.inputs[slotIdx]) {
                        let pos = null;
                        if (typeof outputNode.getConnectionPos === "function") {
                            pos = outputNode.getConnectionPos(true, slotIdx);
                        } else {
                            pos = LGraphNode.prototype.getConnectionPos.call(outputNode, true, slotIdx);
                        }
                        if (pos && !isNaN(pos[0])) {
                            finalPos = pos;
                        }
                    }
                }
            } catch (err) {
                console.warn("[SlotJumper] 原生 outputNode.getConnectionPos 异常，切换至数学几何逆解:", err);
            }
            
            // 2. 尝试从 Subgraph.outputs 集合提取真实记录坐标
            if (!finalPos) {
                try {
                    const outputs = graph.outputs || graph._outputs;
                    const list = outputs instanceof Map ? Array.from(outputs.values()) : (Array.isArray(outputs) ? outputs : Object.values(outputs || {}));
                    const subItem = list.find(i => (i.name || i.label) === outName);
                    if (subItem && subItem.pos && !isNaN(subItem.pos[0])) {
                        finalPos = subItem.pos;
                    }
                } catch (e) {}
            }
            
            // 3. 【核心救命兜底】：中点向量正向推算（终点 = 2 * 中点 - 起点）
            // 绝不依赖任何原生易崩属性，纯数学求解绝对精度坐标
            if (!finalPos && link && link._pos && !isNaN(link._pos[0])) {
                try {
                    const startPos = innerNode.getConnectionPos(false, innerSlotIdx);
                    if (startPos && !isNaN(startPos[0])) {
                        const endX = 2 * link._pos[0] - startPos[0];
                        const endY = 2 * link._pos[1] - startPos[1];
                        finalPos = [endX, endY];
                    }
                } catch (e) {}
            }
            
            // 4. 执行居中跳转
            if (finalPos) {
                jumpToPos(finalPos);
            } else {
                // 终极安全退避
                const p = innerNode.getConnectionPos(false, innerSlotIdx);
                jumpToPos([p[0] + 300, p[1]]);
            }
        }

        // 跳转到外联输入端口 (中点反向推导起点)
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
        // =========================================================================
        // 高亮双环波纹动画 (原版 Canvas 双脉冲方案)
        // =========================================================================
        const origDrawForeground = app.canvas.onDrawForeground;
        app.canvas.onDrawForeground = function (ctx) {
            origDrawForeground?.apply(this, arguments);
            
            if (highlightData) {
                const elapsed = performance.now() - animStartTime;
                
                // 1. 【控制消失时间】：动画总持续时间 (毫秒，1500 = 1.5 秒，可自行调节)
                const duration = 1200;
                
                if (elapsed > duration) {
                    highlightData = null;
                } else {
                    const progress = elapsed / duration; // 进度：从 0 匀速到 1
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
                        const alpha = Math.max(0, 1 - progress); // 渐隐透明度
                        
                        // ---------------------------------------------------------
                        // 环 1：内环，从 6 扩展到 64
                        // ---------------------------------------------------------
                        ctx.beginPath();
                        const radius1 = 6 + progress * (64 - 6);
                        ctx.arc(pos[0], pos[1], radius1, 0, Math.PI * 2);
                        ctx.strokeStyle = `rgba(0, 230, 255, ${alpha})`;
                        ctx.lineWidth = 3;
                        ctx.stroke();
                        
                        // ---------------------------------------------------------
                        // 环 2：外环，从 6 扩展到 128 (速度更快，范围更广)
                        // ---------------------------------------------------------
                        ctx.beginPath();
                        const radius2 = 6 + progress * (128 - 6);
                        ctx.arc(pos[0], pos[1], radius2, 0, Math.PI * 2);
                        // 外环透明度略减 (alpha * 0.7)，营造远景冲击波层次感
                        ctx.strokeStyle = `rgba(0, 230, 255, ${alpha})`;
                        ctx.lineWidth = 2;
                        ctx.stroke();
                        
                        // ---------------------------------------------------------
                        // 中心实心圆点 (随时间同步淡出)
                        // ---------------------------------------------------------
                        ctx.beginPath();
                        ctx.arc(pos[0], pos[1], 7, 0, Math.PI * 2);
                        ctx.fillStyle = "#00e6ff";
                        ctx.shadowColor = "#00e6ff";
                        ctx.shadowBlur = 15;
                        ctx.fill();
                        
                        ctx.restore();
                        
                        // 保持每帧刷新直到动画结束
                        app.canvas.setDirty(true, false);
                    }
                }
            }
        };
    }
});