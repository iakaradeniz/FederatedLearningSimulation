const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  // Server
  startServer: (config) => ipcRenderer.invoke("start-server", config),
  
  // Clients
  startClient: (clientId, config) => ipcRenderer.invoke("start-client", clientId, config),
  
  // Process control
  stopProcess: (processId) => ipcRenderer.invoke("stop-process", processId),
  getProcessStatus: (processId) => ipcRenderer.invoke("get-process-status", processId),
  getAllStatus: () => ipcRenderer.invoke("get-all-status"),
  getPythonDir: () => ipcRenderer.invoke("get-python-dir"),

  // Event listeners
  onProcessLog: (callback) => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on("process-log", handler);
    return () => ipcRenderer.removeListener("process-log", handler);
  },
  
  onProcessStatus: (callback) => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on("process-status", handler);
    return () => ipcRenderer.removeListener("process-status", handler);
  },
  
  onMetricUpdate: (callback) => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on("metric-update", handler);
    return () => ipcRenderer.removeListener("metric-update", handler);
  },

  // Window controls
  minimize: () => ipcRenderer.invoke("window-minimize"),
  maximize: () => ipcRenderer.invoke("window-maximize"),
  close: () => ipcRenderer.invoke("window-close"),
  runInference: (params) => ipcRenderer.invoke("run-inference", params),
  selectFile:   (opts)   => ipcRenderer.invoke("select-file", opts),

});
