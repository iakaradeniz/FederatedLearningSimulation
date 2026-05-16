import { BarChart3 } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

const COLORS = ["#3b82f6", "#10b981", "#f59e0b"];

export default function MetricsPanel({ state, hospitalNames }) {
  const { metrics } = state;
  const serverM = metrics.server;

  // Server validation data
  const valData = serverM.valAcc.map((acc, i) => ({
    round: i + 1,
    accuracy: (acc * 100).toFixed(2),
    loss: serverM.valLoss[i]?.toFixed(4),
  }));

  // Client comparison data - align by round
  const maxRounds = Math.max(
    metrics.client_0.trainLoss.length,
    metrics.client_1.trainLoss.length,
    metrics.client_2.trainLoss.length
  );

  const clientLossData = [];
  const clientAccData = [];
  for (let i = 0; i < maxRounds; i++) {
    const lossRow = { round: i + 1 };
    const accRow = { round: i + 1 };
    [0, 1, 2].forEach((c) => {
      const cm = metrics[`client_${c}`];
      if (cm.trainLoss[i] !== undefined) lossRow[hospitalNames[c]] = cm.trainLoss[i].toFixed(4);
      if (cm.trainAcc[i] !== undefined) accRow[hospitalNames[c]] = (cm.trainAcc[i] * 100).toFixed(2);
    });
    clientLossData.push(lossRow);
    clientAccData.push(accRow);
  }

  const tooltipStyle = { background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 };
  const hasData = valData.length > 0 || maxRounds > 0;

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2><BarChart3 size={24} style={{ display: "inline", marginRight: 8 }} />Metrikler</h2>
        <p>Tüm eğitim metriklerinin detaylı analizi</p>
      </div>

      {!hasData && (
        <div className="card" style={{ textAlign: "center", padding: 60 }}>
          <BarChart3 size={48} style={{ color: "var(--text-muted)", marginBottom: 16 }} />
          <h3 style={{ color: "var(--text-secondary)", marginBottom: 8 }}>Henüz Veri Yok</h3>
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>Eğitim başladığında metrikler burada görünecek.</p>
        </div>
      )}

      {valData.length > 0 && (
        <>
          <div className="section-title">Sunucu — Global Model Validasyonu</div>
          <div className="charts-grid" style={{ marginBottom: 24 }}>
            <div className="card">
              <div className="card-header"><h3>Validasyon Doğruluğu (%)</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={valData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="round" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot={false} name="Val Accuracy" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card">
              <div className="card-header"><h3>Validasyon Kaybı</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={valData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="round" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} name="Val Loss" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}

      {maxRounds > 0 && (
        <>
          <div className="section-title">İstemci Karşılaştırması</div>
          <div className="charts-grid" style={{ marginBottom: 24 }}>
            <div className="card">
              <div className="card-header"><h3>Eğitim Kaybı — Tüm Hastaneler</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={clientLossData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="round" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend />
                    {hospitalNames.map((name, i) => (
                      <Line key={i} type="monotone" dataKey={name} stroke={COLORS[i]} strokeWidth={2} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card">
              <div className="card-header"><h3>Eğitim Doğruluğu — Tüm Hastaneler</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={clientAccData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="round" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend />
                    {hospitalNames.map((name, i) => (
                      <Line key={i} type="monotone" dataKey={name} stroke={COLORS[i]} strokeWidth={2} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
