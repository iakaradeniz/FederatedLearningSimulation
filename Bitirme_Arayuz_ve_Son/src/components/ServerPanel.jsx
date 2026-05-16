import { Play, Square, Server } from "lucide-react";
import StatusBadge from "./StatusBadge";
import LogViewer from "./LogViewer";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function ServerPanel({ state, dispatch, onStart, onStop }) {
  const { statuses, logs, metrics, config } = state;
  const status = statuses.server;
  const m = metrics.server;
  const isRunning = status === "running";

  const chartData = m.valAcc.map((acc, i) => ({
    round: i + 1,
    accuracy: (acc * 100).toFixed(2),
    loss: m.valLoss[i]?.toFixed(4),
  }));

  const setConfig = (key, val) => dispatch({ type: "SET_CONFIG", config: { [key]: val } });

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Sunucu Yönetimi</h2>
        <p>gRPC sunucu konfigürasyonu ve kontrol paneli</p>
      </div>

      {/* Config */}
      <div className="config-panel">
        <div className="config-panel-title">Sunucu Ayarları</div>
        <div className="form-grid">

          {/* YENİ EKLENEN KISIM: Ağ Modu Seçimi */}
          <div className="form-group">
            <label>Ağ Modu (Host)</label>
            <select
              className="form-select"
              value={config.host || "127.0.0.1"}
              onChange={(e) => setConfig("host", e.target.value)}
              disabled={isRunning}
            >
              <option value="127.0.0.1">Yerel (Localhost - 127.0.0.1)</option>
              <option value="0.0.0.0">Dışa Açık (Tüm Ağ - 0.0.0.0)</option>
            </select>
          </div>
          {/* ---------------------------------- */}

          <div className="form-group">
            <label>Port</label>
            <input className="form-input" type="number" value={config.port} onChange={(e) => setConfig("port", parseInt(e.target.value))} disabled={isRunning} />
          </div>
          <div className="form-group">
            <label>Min. İstemci</label>
            <input className="form-input" type="number" value={config.minClients} onChange={(e) => setConfig("minClients", parseInt(e.target.value))} disabled={isRunning} />
          </div>
          <div className="form-group">
            <label>Toplam Tur</label>
            <input className="form-input" type="number" value={config.rounds} onChange={(e) => setConfig("rounds", parseInt(e.target.value))} disabled={isRunning} />
          </div>
          <div className="form-group">
            <label>Model</label>
            <select className="form-select" value={config.model} onChange={(e) => setConfig("model", e.target.value)} disabled={isRunning}>
              <option value="ResNet18">ResNet18</option>
              <option value="MobileNetV2">MobileNetV2</option>
              <option value="DenseNet121">DenseNet121</option>
              <option value="MobileViT">MobileViT</option>
            </select>
          </div>
          <div className="form-group">
            <label>Batch Size</label>
            <input className="form-input" type="number" value={config.batchSize} onChange={(e) => setConfig("batchSize", parseInt(e.target.value))} disabled={isRunning} />
          </div>
          <div className="form-group">
            <label>Veri Dizini</label>
            <input className="form-input" type="text" value={config.dataDir} onChange={(e) => setConfig("dataDir", e.target.value)} disabled={isRunning} />
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 20 }}>
        <button className="btn btn-primary" onClick={onStart} disabled={isRunning}><Play size={16} /> Sunucuyu Başlat</button>
        <button className="btn btn-danger" onClick={onStop} disabled={!isRunning}><Square size={16} /> Durdur</button>
        <StatusBadge status={status} />
        {m.currentRound > 0 && <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>Tur: {m.currentRound}/{m.totalRounds}</span>}
        {m.bestAcc > 0 && <span style={{ fontSize: 13, color: "var(--accent-green)" }}>En İyi: %{(m.bestAcc * 100).toFixed(2)}</span>}
      </div>

      {/* Charts */}
      {chartData.length > 0 && (
        <div className="charts-grid" style={{ marginBottom: 20 }}>
          <div className="card">
            <div className="card-header"><h3>Validasyon Doğruluğu</h3></div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="round" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                  <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot={false} name="Accuracy (%)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card">
            <div className="card-header"><h3>Validasyon Kaybı</h3></div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="round" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                  <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} name="Loss" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Log */}
      <div className="card">
        <div className="card-header"><h3><Server size={16} /> Sunucu Logları</h3></div>
        <div className="card-body"><LogViewer logs={logs.server} /></div>
      </div>
    </div>
  );
}