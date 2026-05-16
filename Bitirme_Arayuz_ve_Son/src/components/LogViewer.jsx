import { useRef, useEffect } from "react";

export default function LogViewer({ logs = [] }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  if (logs.length === 0) {
    return (
      <div className="log-viewer" style={{ color: "var(--text-muted)", fontStyle: "italic", display: "flex", alignItems: "center", justifyContent: "center", minHeight: 120 }}>
        Henüz log yok. İşlemi başlatın...
      </div>
    );
  }

  return (
    <div className="log-viewer">
      {logs.map((log, i) => (
        <div key={i} className="log-line">
          <span className="log-time">{new Date(log.timestamp).toLocaleTimeString("tr-TR")}</span>
          <span className={`log-text ${log.type}`}>{log.data}</span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
