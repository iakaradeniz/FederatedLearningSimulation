export default function StatusBadge({ status, mini }) {
  const labels = { idle: "Bekliyor", running: "Çalışıyor", starting: "Başlatılıyor", completed: "Tamamlandı", error: "Hata" };
  if (mini) {
    return (
      <span className={`status-badge ${status}`} style={{ padding: "2px 6px", fontSize: "9px" }}>
        {labels[status] || status}
      </span>
    );
  }
  return <span className={`status-badge ${status}`}>{labels[status] || status}</span>;
}
