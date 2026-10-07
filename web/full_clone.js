import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "lhyNodes.FullClone",
    async setup() {
        // 拦截底层的节点右键菜单构建方法
        const origGetNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;

        LGraphCanvas.prototype.getNodeMenuOptions = function(node) {
            // 获取原生菜单列表（包含 Clone、Remove 等）
            const options = origGetNodeMenuOptions.apply(this, arguments);

            // 避免重复添加
            const alreadyExists = options.some(
                opt => opt && opt.content === "Clone (with inputs)"
            );
            if (alreadyExists) return options;

            // 构造我们的新菜单项
            const fullCloneOption = {
                content: "Clone (with inputs)",
                callback: () => {
                    fullCloneNode(node);
                }
            };

            // 查找原版 "Clone"（同时兼容英文原版与中文汉化版）所在索引
            const cloneIndex = options.findIndex(opt => {
                if (!opt) return false;
                if (typeof opt === "string") {
                    const text = opt.trim().toLowerCase();
                    return text === "clone";
                }
                if (typeof opt.content === "string") {
                    const text = opt.content.trim().toLowerCase();
                    return text === "clone";
                }
                return false;
            });

            if (cloneIndex !== -1) {
                options.splice(cloneIndex + 1, 0, fullCloneOption);
            } else {
                options.push(fullCloneOption);
            }

            return options;
        };
    }
});

/**
 * 完整克隆逻辑
 */
function fullCloneNode(originalNode) {
    if (!originalNode || !originalNode.graph) return;

    const graph = originalNode.graph;

    // 1. 完整序列化原节点状态
    const nodeData = JSON.parse(JSON.stringify(originalNode.serialize()));
    delete nodeData.id;

    if (nodeData.inputs) {
        nodeData.inputs.forEach(input => input.link = null);
    }
    if (nodeData.outputs) {
        nodeData.outputs.forEach(output => output.links = null);
    }

    // 2. 实例化并配置新节点
    const newNode = LiteGraph.createNode(originalNode.type);
    if (!newNode) return;

    newNode.configure(nodeData);

    // 3. 错位放置
    newNode.pos = [
        originalNode.pos[0] + 40,
        originalNode.pos[1] + 40
    ];

    graph.add(newNode);

    // 4. 重建上游输入连线
    if (originalNode.inputs && originalNode.inputs.length > 0) {
        for (let i = 0; i < originalNode.inputs.length; i++) {
            const linkId = originalNode.inputs[i].link;
            if (linkId !== null && linkId !== undefined) {
                const link = graph.links[linkId];
                if (link) {
                    const originNode = graph.getNodeById(link.origin_id);
                    if (originNode) {
                        originNode.connect(link.origin_slot, newNode, i);
                    }
                }
            }
        }
    }

    // 5. 选中新节点
    if (app.canvas) {
        app.canvas.selectNode(newNode);
    }
    graph.setDirtyCanvas(true, true);
}