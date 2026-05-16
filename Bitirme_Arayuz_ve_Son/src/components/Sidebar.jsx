import { LayoutDashboard, Server, Building2, BarChart3 } from "lucide-react";
import StatusBadge from "./StatusBadge";
import { Microscope } from "lucide-react";

const HOSPITAL_COLORS = ["var(--accent-blue)", "var(--accent-green)", "var(--accent-orange)"];

export default function Sidebar({ page, setPage, statuses, hospitalNames }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">FL</div>
        <div className="sidebar-logo-text">
          <h1>FedLearn</h1>
          <span>Yönetim Paneli</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <button className={`nav-item ${page === "dashboard" ? "active" : ""}`} onClick={() => setPage("dashboard")}>
          <LayoutDashboard size={18} /> Dashboard
        </button>

        <div className="nav-section-title">Sunucu</div>
        <button className={`nav-item ${page === "server" ? "active" : ""}`} onClick={() => setPage("server")}>
          <Server size={18} />
          <span style={{ flex: 1 }}>Sunucu</span>
          <StatusBadge status={statuses.server} mini />
        </button>

        <div className="nav-section-title">İstemciler</div>
        {[0, 1, 2].map((i) => (
          <button key={i} className={`nav-item ${page === `client_${i}` ? "active" : ""}`} onClick={() => setPage(`client_${i}`)}>
            <Building2 size={18} style={{ color: HOSPITAL_COLORS[i] }} />
            <span style={{ flex: 1 }}>{hospitalNames[i]}</span>
            <StatusBadge status={statuses[`client_${i}`]} mini />
          </button>
        ))}

        <div className="nav-section-title">Analiz</div>
        <button className={`nav-item ${page === "metrics" ? "active" : ""}`} onClick={() => setPage("metrics")}>
          <BarChart3 size={18} /> Metrikler
        </button>
        <button className={`nav-item ${page === "inference" ? "active" : ""}`} onClick={() => setPage("inference")}>
          <Microscope size={18} /> Inference
        </button>

      </nav>
    </aside>
  );
}
