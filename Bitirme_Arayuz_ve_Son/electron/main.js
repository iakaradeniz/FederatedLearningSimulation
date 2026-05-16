const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const { spawn } = require("child_process");


// ─── Globals ────────────────────────────────────────────────
let mainWindow = null;

// Managed subprocesses: { "server": ChildProcess, "client_0": ChildProcess, ... }
const processes = {};

// Python scripts directory (adim3)
const PYTHON_DIR = path.resolve(__dirname, "../../Bitirme-Calismasi/adim3");

function getPythonPath() {
  if (app.isPackaged) {
    // Örnek: uygulama ile paketlenmiş Python
    return path.join(process.resourcesPath, "python", "python.exe");
  }
  // Geliştirme: sistemdeki python3 / python
  return process.platform === "win32" ? "python" : "python3";
}
 
function getInferenceScriptPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "inference.py");
  }
  // Dev: proje kökündeki inference.py
  return path.join(__dirname, "..", "inference.py");
}

// ─── Window Creation ────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    frame: false,
    titleBarStyle: "hidden",
    backgroundColor: "#0a0e1a",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Dev or prod
  const isDev = process.env.NODE_ENV === "development" || process.argv.includes("--dev");
  if (isDev) {
    mainWindow.loadURL("http://localhost:5173");
  } else {
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ─── Helper: Send log to renderer ────────────────────────────
function sendLog(processId, type, data) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("process-log", {
      processId,
      type, // "stdout" | "stderr" | "system"
      data: data.toString(),
      timestamp: Date.now(),
    });
  }
}

// ─── Helper: Send status update ──────────────────────────────
function sendStatus(processId, status) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("process-status", { processId, status });
  }
}

// ─── Helper: Parse metrics from log lines ────────────────────
function parseMetrics(processId, line) {
  if (!mainWindow || mainWindow.isDestroyed()) return;

  // Server validation metrics:
  //   "Validasyon | Val Loss: 0.1234 | Val Acc: 0.8765"
  const valMatch = line.match(/Val Loss:\s*([\d.]+)\s*\|\s*Val Acc:\s*([\d.]+)/);
  if (valMatch) {
    mainWindow.webContents.send("metric-update", {
      processId,
      type: "validation",
      valLoss: parseFloat(valMatch[1]),
      valAcc: parseFloat(valMatch[2]),
      timestamp: Date.now(),
    });
  }

  // Server round completion:
  //   "Tur 5/75 tamamlandı: FedAvg uygulandı (3 istemci)"
  const roundMatch = line.match(/Tur\s+(\d+)\/(\d+)\s+tamamlandı/);
  if (roundMatch) {
    mainWindow.webContents.send("metric-update", {
      processId,
      type: "round",
      currentRound: parseInt(roundMatch[1]),
      totalRounds: parseInt(roundMatch[2]),
      timestamp: Date.now(),
    });
  }

  // Client training metrics:
  //   "Train Loss: 0.1234 | Train Acc: 0.8765"
  const trainMatch = line.match(/Train Loss:\s*([\d.]+)\s*\|\s*Train Acc:\s*([\d.]+)/);
  if (trainMatch) {
    mainWindow.webContents.send("metric-update", {
      processId,
      type: "training",
      trainLoss: parseFloat(trainMatch[1]),
      trainAcc: parseFloat(trainMatch[2]),
      timestamp: Date.now(),
    });
  }

  // Client round info:
  //   "[ Federated Tur 5/75 ]"
  const clientRoundMatch = line.match(/Federated Tur\s+(\d+)\/(\d+)/);
  if (clientRoundMatch) {
    mainWindow.webContents.send("metric-update", {
      processId,
      type: "clientRound",
      currentRound: parseInt(clientRoundMatch[1]),
      totalRounds: parseInt(clientRoundMatch[2]),
      timestamp: Date.now(),
    });
  }

  // Best model saved:
  //   "★ Yeni en iyi model! Val Acc: 0.9123"
  const bestMatch = line.match(/Yeni en iyi model.*Val Acc:\s*([\d.]+)/);
  if (bestMatch) {
    mainWindow.webContents.send("metric-update", {
      processId,
      type: "bestModel",
      bestAcc: parseFloat(bestMatch[1]),
      timestamp: Date.now(),
    });
  }

  // All rounds completed
  if (line.includes("TÜM TURLAR TAMAMLANDI") || line.includes("Federated Learning tamamlandı")) {
    mainWindow.webContents.send("metric-update", {
      processId,
      type: "completed",
      timestamp: Date.now(),
    });
  }
}

// ─── Spawn a Python process ─────────────────────────────────
function spawnPython(processId, scriptName, args) {
  if (processes[processId]) {
    sendLog(processId, "system", "⚠ İşlem zaten çalışıyor. Önce durdurun.");
    return false;
  }

  const scriptPath = path.join(PYTHON_DIR, scriptName);

  sendLog(processId, "system", `🚀 Başlatılıyor: python ${scriptName} ${args.join(" ")}`);
  sendStatus(processId, "starting");

  const child = spawn("python", [scriptPath, ...args], {
    cwd: PYTHON_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  processes[processId] = child;

  child.stdout.on("data", (data) => {
    const lines = data.toString().split("\n");
    lines.forEach((line) => {
      if (line.trim()) {
        sendLog(processId, "stdout", line);
        parseMetrics(processId, line);
      }
    });
  });

  child.stderr.on("data", (data) => {
    const lines = data.toString().split("\n");
    lines.forEach((line) => {
      if (line.trim()) {
        sendLog(processId, "stderr", line);
        parseMetrics(processId, line);
      }
    });
  });

  child.on("error", (err) => {
    sendLog(processId, "system", `❌ Hata: ${err.message}`);
    sendStatus(processId, "error");
    delete processes[processId];
  });

  child.on("close", (code) => {
    sendLog(processId, "system", `✅ İşlem sonlandı (kod: ${code})`);
    sendStatus(processId, code === 0 ? "completed" : "error");
    delete processes[processId];
  });

  sendStatus(processId, "running");
  return true;
}

// ─── IPC Handlers ────────────────────────────────────────────
ipcMain.handle("start-server", (_event, config) => {
  const args = [
    "--port", String(config.port || 50051),
    "--min-clients", String(config.minClients || 3),
    "--rounds", String(config.rounds || 75),
    "--model", config.model || "MobileNetV2",
    "--data-dir", config.dataDir || "./Federated_Dataset_Yeni/Federated_Dataset",
    "--batch-size", String(config.batchSize || 32),
  ];
  return spawnPython("server", "server.py", args);
});

ipcMain.handle("start-client", (_event, clientId, config) => {
  const processId = `client_${clientId}`;
  const args = [
    "--client-id", String(clientId),
    "--server", config.serverAddr || "100.77.33.86:50051",
    "--model", config.model || "MobileNetV2",
    "--rounds", String(config.rounds || 75),
    "--local-epochs", String(config.localEpochs || 1),
    "--lr", String(config.lr || 0.001),
    "--batch-size", String(config.batchSize || 32),
    "--fedprox-mu", String(config.fedproxMu || 0.01),
    "--data-dir", config.dataDir || "./Federated_Dataset_Yeni/Federated_Dataset",
  ];
  return spawnPython(processId, "client.py", args);
});

ipcMain.handle("stop-process", (_event, processId) => {
  const child = processes[processId];
  if (child) {
    sendLog(processId, "system", "🛑 İşlem durduruluyor...");
    child.kill("SIGTERM");
    setTimeout(() => {
      if (processes[processId]) {
        processes[processId].kill("SIGKILL");
        delete processes[processId];
      }
    }, 3000);
    return true;
  }
  return false;
});

ipcMain.handle("get-process-status", (_event, processId) => {
  return processes[processId] ? "running" : "idle";
});

ipcMain.handle("get-all-status", () => {
  const status = {};
  for (const key of ["server", "client_0", "client_1", "client_2"]) {
    status[key] = processes[key] ? "running" : "idle";
  }
  return status;
});

ipcMain.handle("get-python-dir", () => PYTHON_DIR);

// Window controls
ipcMain.handle("window-minimize", () => mainWindow?.minimize());
ipcMain.handle("window-maximize", () => {
  if (mainWindow?.isMaximized()) mainWindow.unmaximize();
  else mainWindow?.maximize();
});
ipcMain.handle("window-close", () => mainWindow?.close());

ipcMain.handle("select-file", async (_event, options = {}) => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: options.filters || [
      { name: "Tüm Dosyalar", extensions: ["*"] }
    ],
  });
  return canceled ? null : filePaths[0];
});
 
// -----------------------------------------------------------------------
// IPC: Inference çalıştırma
// -----------------------------------------------------------------------
ipcMain.handle("run-inference", async (_event, { imagePath, modelPath, arch }) => {
  return new Promise((resolve) => {
    const python = getPythonPath();
    const script = getInferenceScriptPath();
 
    const args = [
      script,
      "--image",      imagePath,
      "--model-path", modelPath,
      "--arch",       arch,
      "--top-k",      "5",
      "--device",     "auto",
    ];
 
    let stdout = "";
    let stderr = "";
 
    const proc = spawn(python, args, { stdio: ["ignore", "pipe", "pipe"] });
 
    proc.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    proc.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
 
    proc.on("close", (code) => {
      if (code !== 0 && !stdout) {
        resolve({
          success: false,
          error: `Python çıkış kodu ${code}.\n${stderr.slice(0, 500)}`,
        });
        return;
      }
 
      try {
        // inference.py her zaman son satırda JSON basar
        const lines  = stdout.trim().split("\n");
        const jsonLine = lines[lines.length - 1];
        const result = JSON.parse(jsonLine);
        resolve(result);
      } catch (e) {
        resolve({
          success: false,
          error: `JSON parse hatası: ${e.message}\nStdout: ${stdout.slice(0, 300)}`,
        });
      }
    });
 
    proc.on("error", (err) => {
      resolve({
        success: false,
        error: `Python başlatılamadı: ${err.message}\nKontrol: python yüklü mü, PATH'te mi?`,
      });
    });
  });
});


// ─── App Lifecycle ───────────────────────────────────────────
app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  // Kill all subprocesses
  Object.keys(processes).forEach((id) => {
    try { processes[id].kill(); } catch (e) { /* ignore */ }
  });
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
