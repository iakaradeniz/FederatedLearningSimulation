import { Play, Square, Building2 } from "lucide-react";
import StatusBadge from "./StatusBadge";
import LogViewer from "./LogViewer";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const COLORS = ["#f6673bff", "#10b981", "#f59e0b"];

export default function ClientPanel({ clientId, hospitalName, state, dispatch, onStart, onStop }) {
  const processId = `client_${clientId}`;
  const { statuses, logs, metrics, config } = state;
  const status = statuses[processId];
  const m = metrics[processId];
  const isRunning = status === "running";
  const color = COLORS[clientId];

  const lossData = m.trainLoss.map((l, i) => ({ round: i + 1, loss: l.toFixed(4) }));
  const accData = m.trainAcc.map((a, i) => ({ round: i + 1, accuracy: (a * 100).toFixed(2) }));

  const setConfig = (key, val) => dispatch({ type: "SET_CONFIG", config: { [key]: val } });

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2 style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Building2 size={24} style={{ color }} /> {hospitalName}
        </h2>
        <p>İstemci {clientId} — gRPC federatif öğrenme istemcisi</p>
      </div>

      {/* Config */}
      <div className="config-panel">
        <div className="config-panel-title">İstemci Ayarları</div>
        <div className="form-grid">
          <div className="form-group">
            <label>Sunucu Adresi</label>
            <input className="form-input" type="text" value={config.serverAddr} onChange={(e) => setConfig("serverAddr", e.target.value)} disabled={isRunning} />
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
            <label>Toplam Tur</label>
            <input className="form-input" type="number" value={config.rounds} onChange={(e) => setConfig("rounds", parseInt(e.target.value))} disabled={isRunning} />
          </div>
          <div className="form-group">
            <label>Öğrenme Hızı</label>
            <input className="form-input" type="number" step="0.0001" value={config.lr} onChange={(e) => setConfig("lr", parseFloat(e.target.value))} disabled={isRunning} />
          </div>
          <div className="form-group">
            <label>Batch Size</label>
            <input className="form-input" type="number" value={config.batchSize} onChange={(e) => setConfig("batchSize", parseInt(e.target.value))} disabled={isRunning} />
          </div>
          <div className="form-group">
            <label>FedProx μ</label>
            <input className="form-input" type="number" step="0.001" value={config.fedproxMu} onChange={(e) => setConfig("fedproxMu", parseFloat(e.target.value))} disabled={isRunning} />
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 20 }}>
        <button className="btn btn-primary" onClick={onStart} disabled={isRunning}><Play size={16} /> Başlat</button>
        <button className="btn btn-danger" onClick={onStop} disabled={!isRunning}><Square size={16} /> Durdur</button>
        <StatusBadge status={status} />
        {m.currentRound > 0 && <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>Tur: {m.currentRound}/{m.totalRounds}</span>}
      </div>

      {/* Metrics Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
        <div className="metric-item" style={{ background: "var(--bg-card)", border: "1px solid var(--border-glass)", borderRadius: "var(--radius)" }}>
          <div className="metric-value" style={{ color }}>{m.trainLoss.length > 0 ? m.trainLoss[m.trainLoss.length - 1].toFixed(4) : "—"}</div>
          <div className="metric-label">Son Train Loss</div>
        </div>
        <div className="metric-item" style={{ background: "var(--bg-card)", border: "1px solid var(--border-glass)", borderRadius: "var(--radius)" }}>
          <div className="metric-value" style={{ color }}>{m.trainAcc.length > 0 ? `%${(m.trainAcc[m.trainAcc.length - 1] * 100).toFixed(1)}` : "—"}</div>
          <div className="metric-label">Son Train Acc</div>
        </div>
        <div className="metric-item" style={{ background: "var(--bg-card)", border: "1px solid var(--border-glass)", borderRadius: "var(--radius)" }}>
          <div className="metric-value">{m.currentRound || 0}</div>
          <div className="metric-label">Mevcut Tur</div>
        </div>
        <div className="metric-item" style={{ background: "var(--bg-card)", border: "1px solid var(--border-glass)", borderRadius: "var(--radius)" }}>
          <div className="metric-value">{m.trainLoss.length}</div>
          <div className="metric-label">Toplam Epoch</div>
        </div>
      </div>

      {/* Charts */}
      {lossData.length > 0 && (
        <div className="charts-grid" style={{ marginBottom: 20 }}>
          <div className="card">
            <div className="card-header"><h3>Eğitim Kaybı (Loss)</h3></div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="round" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                  <Line type="monotone" dataKey="loss" stroke={color} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card">
            <div className="card-header"><h3>Eğitim Doğruluğu (Accuracy)</h3></div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={accData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="round" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                  <Line type="monotone" dataKey="accuracy" stroke={color} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Log */}
      <div className="card">
        <div className="card-header"><h3><Building2 size={16} style={{ color }} /> {hospitalName} Logları</h3></div>
        <div className="card-body"><LogViewer logs={logs[processId]} /></div>
      </div>
    </div>
  );
}
