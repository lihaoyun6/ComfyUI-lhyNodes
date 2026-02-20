import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const postContinue = (nodeId) => fetch("/lhy_queuehandler/continue/" + nodeId, { method: "POST" });

app.registerExtension({
  name: "lhy-QueueHandler",
  nodeCreated(node) {
    if (node.comfyClass === "QueueHandler") {
      node.addWidget("button", "Continue", "CONTINUE", () => {
          postContinue(node.id);
        });
    }},
});