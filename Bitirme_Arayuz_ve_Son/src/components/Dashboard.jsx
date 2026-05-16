import { Server, Users, Activity, Zap, Play, Square, Rocket } from "lucide-react";
import StatusBadge from "./StatusBadge";
import LogViewer from "./LogViewer";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const COLORS = ["#3b82f6", "#10b981", "#f59e0b"];

export default function Dashboard({ state, onStartServer, onStartClient, onStopProcess, onStartAll, hospitalNames }) {
  const { statuses, metrics, logs } = state;
  const runningCount = Object.values(statuses).filter((s) => s === "running").length;
  const serverM = metrics.server;
  const currentRound = serverM.currentRound || 0;
  const totalRounds = serverM.totalRounds || state.config.rounds;
  const bestAcc = serverM.bestAcc || 0;

  const chartData = serverM.valAcc.map((acc, i) => ({
    round: i + 1,
    accuracy: (acc * 100).toFixed(2),
    loss: serverM.valLoss[i]?.toFixed(4),
  }));

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Federatif öğrenme sisteminin genel durumu</p>
      </div>

      {/* Stats */}
      <div className="dashboard-grid">
        <div className="stat-card">
          <div className="stat-icon blue"><Server size={20} /></div>
          <div className="stat-value">{statuses.server === "running" ? "Aktif" : "Kapalı"}</div>
          <div className="stat-label">Sunucu Durumu</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon green"><Users size={20} /></div>
          <div className="stat-value">{runningCount > 0 ? runningCount - (statuses.server === "running" ? 1 : 0) : 0}/3</div>
          <div className="stat-label">Aktif İstemci</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon orange"><Activity size={20} /></div>
          <div className="stat-value">{currentRound}/{totalRounds}</div>
          <div className="stat-label">Mevcut Tur</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon purple"><Zap size={20} /></div>
          <div className="stat-value">%{(bestAcc * 100).toFixed(1)}</div>
          <div className="stat-label">En İyi Doğruluk</div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        <button className="btn btn-primary" onClick={onStartAll} disabled={statuses.server === "running"}>
          <Rocket size={16} /> Tümünü Başlat
        </button>
        <button className="btn btn-success" onClick={onStartServer} disabled={statuses.server === "running"}>
          <Play size={16} /> Sunucuyu Başlat
        </button>
        {[0, 1, 2].map((i) => (
          <button key={i} className="btn btn-ghost" onClick={() => onStartClient(i)} disabled={statuses[`client_${i}`] === "running"}>
            <Play size={14} /> {hospitalNames[i]}
          </button>
        ))}
        {Object.entries(statuses).some(([, s]) => s === "running") && (
          <button className="btn btn-danger" onClick={() => Object.keys(statuses).forEach((k) => { if (statuses[k] === "running") onStopProcess(k); })}>
            <Square size={14} /> Tümünü Durdur
          </button>
        )}
      </div>

      {/* Client Cards */}
      <div className="section-title">İstemciler (Hastaneler)</div>
      <div className="clients-grid">
        {[0, 1, 2].map((i) => {
          const cm = metrics[`client_${i}`];
          const lastLoss = cm.trainLoss.length > 0 ? cm.trainLoss[cm.trainLoss.length - 1] : null;
          const lastAcc = cm.trainAcc.length > 0 ? cm.trainAcc[cm.trainAcc.length - 1] : null;
          return (
            <div key={i} className={`card client-card hospital-${i}`}>
              <div className="card-header">
                <div className="hospital-name">
                  <div className="hospital-icon">{String.fromCharCode(65 + i)}</div>
                  <div>
                    <h3 style={{ margin: 0 }}>{hospitalNames[i]}</h3>
                    <span style={{ fontSize: 11, color: "var(--text-muted)" }}>client_{i}</span>
                  </div>
                </div>
                <StatusBadge status={statuses[`client_${i}`]} />
              </div>
              <div className="card-body">
                <div className="client-metrics">
                  <div className="metric-item">
                    <div className="metric-value" style={{ color: COLORS[i] }}>{lastLoss !== null ? lastLoss.toFixed(4) : "—"}</div>
                    <div className="metric-label">Train Loss</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-value" style={{ color: COLORS[i] }}>{lastAcc !== null ? `%${(lastAcc * 100).toFixed(1)}` : "—"}</div>
                    <div className="metric-label">Train Acc</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-value">{cm.currentRound || 0}</div>
                    <div className="metric-label">Tur</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-value">{cm.trainLoss.length}</div>
                    <div className="metric-label">Toplam Epoch</div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Validation Chart */}
      {chartData.length > 0 && (
        <div className="card chart-card" style={{ marginBottom: 24 }}>
          <div className="card-header"><h3><Activity size={16} /> Validasyon Doğruluğu</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="round" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Server Log */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header"><h3><Server size={16} /> Sunucu Log</h3></div>
        <div className="card-body"><LogViewer logs={logs.server} /></div>
      </div>
    </div>
  );
}
