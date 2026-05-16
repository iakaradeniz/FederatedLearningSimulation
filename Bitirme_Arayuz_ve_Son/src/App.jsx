import { useState, useReducer, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import ServerPanel from "./components/ServerPanel";
import ClientPanel from "./components/ClientPanel";
import MetricsPanel from "./components/MetricsPanel";
import InferencePanel from "./components/InferencePanel";

const HOSPITAL_NAMES = ["Hastane A", "Hastane B", "Hastane C"];

const initialState = {
  statuses: { server: "idle", client_0: "idle", client_1: "idle", client_2: "idle" },
  logs: { server: [], client_0: [], client_1: [], client_2: [] },
  metrics: {
    server: { valLoss: [], valAcc: [], currentRound: 0, totalRounds: 0, bestAcc: 0 },
    client_0: { trainLoss: [], trainAcc: [], currentRound: 0, totalRounds: 0 },
    client_1: { trainLoss: [], trainAcc: [], currentRound: 0, totalRounds: 0 },
    client_2: { trainLoss: [], trainAcc: [], currentRound: 0, totalRounds: 0 },
  },
  config: {
    port: 50051,
    minClients: 3,
    rounds: 75,
    model: "MobileNetV2",
    dataDir: "./Federated_Dataset_Yeni/Federated_Dataset",
    batchSize: 32,
    serverAddr: "100.77.33.86:50051",
    localEpochs: 1,
    lr: 0.001,
    fedproxMu: 0.01,
  },
};

function reducer(state, action) {
  switch (action.type) {
    case "SET_STATUS": {
      return { ...state, statuses: { ...state.statuses, [action.processId]: action.status } };
    }
    case "ADD_LOG": {
      const logs = { ...state.logs };
      const arr = [...(logs[action.processId] || [])];
      arr.push(action.log);
      if (arr.length > 500) arr.splice(0, arr.length - 500);
      logs[action.processId] = arr;
      return { ...state, logs };
    }
    case "UPDATE_METRIC": {
      const metrics = { ...state.metrics };
      const m = { ...metrics[action.processId] };
      const d = action.data;
      if (d.type === "validation") {
        m.valLoss = [...(m.valLoss || []), d.valLoss];
        m.valAcc = [...(m.valAcc || []), d.valAcc];
      } else if (d.type === "round") {
        m.currentRound = d.currentRound;
        m.totalRounds = d.totalRounds;
      } else if (d.type === "training") {
        m.trainLoss = [...(m.trainLoss || []), d.trainLoss];
        m.trainAcc = [...(m.trainAcc || []), d.trainAcc];
      } else if (d.type === "clientRound") {
        m.currentRound = d.currentRound;
        m.totalRounds = d.totalRounds;
      } else if (d.type === "bestModel") {
        m.bestAcc = d.bestAcc;
      } else if (d.type === "completed") {
        // handled by status
      }
      metrics[action.processId] = m;
      return { ...state, metrics };
    }
    case "SET_CONFIG":
      return { ...state, config: { ...state.config, ...action.config } };
    case "RESET_METRICS": {
      const metrics = { ...state.metrics };
      if (action.processId === "server") {
        metrics.server = { valLoss: [], valAcc: [], currentRound: 0, totalRounds: 0, bestAcc: 0 };
      } else {
        metrics[action.processId] = { trainLoss: [], trainAcc: [], currentRound: 0, totalRounds: 0 };
      }
      return { ...state, metrics };
    }
    default:
      return state;
  }
}

function TitleBar() {
  const api = window.electronAPI;
  return (
    <div className="titlebar">
      <span className="titlebar-title">⚛ Federatif Öğrenme Yönetim Paneli</span>
      <div className="titlebar-controls">
        <button className="titlebar-btn minimize" onClick={() => api?.minimize()} />
        <button className="titlebar-btn maximize" onClick={() => api?.maximize()} />
        <button className="titlebar-btn close" onClick={() => api?.close()} />
      </div>
    </div>
  );
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [page, setPage] = useState("dashboard");

  useEffect(() => {
    const api = window.electronAPI;
    if (!api) return;

    const unsub1 = api.onProcessLog((data) => {
      dispatch({ type: "ADD_LOG", processId: data.processId, log: data });
    });
    const unsub2 = api.onProcessStatus((data) => {
      dispatch({ type: "SET_STATUS", processId: data.processId, status: data.status });
    });
    const unsub3 = api.onMetricUpdate((data) => {
      dispatch({ type: "UPDATE_METRIC", processId: data.processId, data });
    });

    return () => { unsub1(); unsub2(); unsub3(); };
  }, []);

  const handleStartServer = useCallback(async () => {
    const api = window.electronAPI;
    if (!api) return;
    dispatch({ type: "RESET_METRICS", processId: "server" });
    await api.startServer(state.config);
  }, [state.config]);

  const handleStartClient = useCallback(async (clientId) => {
    const api = window.electronAPI;
    if (!api) return;
    dispatch({ type: "RESET_METRICS", processId: `client_${clientId}` });
    await api.startClient(clientId, state.config);
  }, [state.config]);

  const handleStopProcess = useCallback(async (processId) => {
    const api = window.electronAPI;
    if (!api) return;
    await api.stopProcess(processId);
  }, []);

  const handleStartAll = useCallback(async () => {
    await handleStartServer();
    setTimeout(() => handleStartClient(0), 2000);
    setTimeout(() => handleStartClient(1), 3000);
    setTimeout(() => handleStartClient(2), 4000);
  }, [handleStartServer, handleStartClient]);

  const renderPage = () => {
    switch (page) {
      case "dashboard":
        return <Dashboard state={state} onStartServer={handleStartServer} onStartClient={handleStartClient} onStopProcess={handleStopProcess} onStartAll={handleStartAll} hospitalNames={HOSPITAL_NAMES} />;
      case "server":
        return <ServerPanel state={state} dispatch={dispatch} onStart={handleStartServer} onStop={() => handleStopProcess("server")} />;
      case "client_0":
      case "client_1":
      case "client_2": {
        const id = parseInt(page.split("_")[1]);
        return <ClientPanel clientId={id} hospitalName={HOSPITAL_NAMES[id]} state={state} dispatch={dispatch} onStart={() => handleStartClient(id)} onStop={() => handleStopProcess(page)} />;
      }
      case "metrics":
        return <MetricsPanel state={state} hospitalNames={HOSPITAL_NAMES} />;
      
      case "inference":
        return <InferencePanel />;
      default:
        return <Dashboard state={state} onStartServer={handleStartServer} onStartClient={handleStartClient} onStopProcess={handleStopProcess} onStartAll={handleStartAll} hospitalNames={HOSPITAL_NAMES} />;
      
    }
  };

  return (
    <>
      <TitleBar />
      <div className="app-layout">
        <Sidebar page={page} setPage={setPage} statuses={state.statuses} hospitalNames={HOSPITAL_NAMES} />
        <main className="main-content">{renderPage()}</main>
      </div>
    </>
  );
}
