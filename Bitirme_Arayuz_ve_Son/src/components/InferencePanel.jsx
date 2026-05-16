import { useState, useRef, useCallback } from "react";
import { Upload, Microscope, Cpu, ChevronDown, X, AlertCircle, CheckCircle2, Loader2, FlaskConical } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

// -----------------------------------------------------------------------
// Sabitler
// -----------------------------------------------------------------------
const MODELS = [
  { id: "ResNet18",     label: "ResNet-18",      desc: "Hızlı, hafif",                   color: "#3b82f6" },
  { id: "MobileNetV2",  label: "MobileNet V2",   desc: "Mobil optimizasyonlu",            color: "#10b981" },
  { id: "DenseNet121",  label: "DenseNet-121",   desc: "Yoğun bağlantılı",               color: "#f59e0b" },
  { id: "MobileViT",    label: "MobileViT-S",    desc: "Hibrit CNN + Transformer",       color: "#a855f7" },
];

const BAR_COLORS = [
  "#10b981", "#3b82f6", "#f59e0b", "#a855f7", "#ef4444"
];

const CONFIDENCE_THRESHOLDS = {
  high:   { min: 0.80, color: "#10b981", label: "Yüksek" },
  medium: { min: 0.50, color: "#f59e0b", label: "Orta"   },
  low:    { min: 0.00, color: "#ef4444", label: "Düşük"  },
};

function getConfidenceLevel(score) {
  if (score >= CONFIDENCE_THRESHOLDS.high.min)   return CONFIDENCE_THRESHOLDS.high;
  if (score >= CONFIDENCE_THRESHOLDS.medium.min) return CONFIDENCE_THRESHOLDS.medium;
  return CONFIDENCE_THRESHOLDS.low;
}

// -----------------------------------------------------------------------
// Model Seçici
// -----------------------------------------------------------------------
function ModelSelector({ selected, onSelect, modelPath, onModelPathChange }) {
  const [open, setOpen] = useState(false);
  const sel = MODELS.find(m => m.id === selected) || MODELS[0];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Mimari dropdown */}
      <div style={{ position: "relative" }}>
        <div
          onClick={() => setOpen(o => !o)}
          style={{
            display: "flex", alignItems: "center", gap: 12,
            padding: "12px 16px",
            background: "var(--bg-secondary)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 10, cursor: "pointer",
            transition: "border-color 0.2s",
          }}
          onMouseEnter={e => e.currentTarget.style.borderColor = sel.color}
          onMouseLeave={e => e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"}
        >
          <div style={{
            width: 10, height: 10, borderRadius: "50%",
            background: sel.color, flexShrink: 0,
            boxShadow: `0 0 8px ${sel.color}88`
          }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: "var(--text-primary)" }}>{sel.label}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{sel.desc}</div>
          </div>
          <ChevronDown size={16} style={{ color: "var(--text-muted)", transition: "transform 0.2s", transform: open ? "rotate(180deg)" : "none" }} />
        </div>

        {open && (
          <div style={{
            position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, zIndex: 50,
            background: "var(--bg-secondary)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 10, overflow: "hidden",
            boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
          }}>
            {MODELS.map(m => (
              <div
                key={m.id}
                onClick={() => { onSelect(m.id); setOpen(false); }}
                style={{
                  display: "flex", alignItems: "center", gap: 12,
                  padding: "11px 16px", cursor: "pointer",
                  background: selected === m.id ? `${m.color}18` : "transparent",
                  transition: "background 0.15s",
                }}
                onMouseEnter={e => { if (selected !== m.id) e.currentTarget.style.background = "rgba(255,255,255,0.04)"; }}
                onMouseLeave={e => { e.currentTarget.style.background = selected === m.id ? `${m.color}18` : "transparent"; }}
              >
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: m.color, flexShrink: 0 }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>{m.label}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{m.desc}</div>
                </div>
                {selected === m.id && <CheckCircle2 size={14} style={{ color: m.color }} />}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* .pth dosya yolu */}
      <div>
        <label style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: 6 }}>
          Model Dosyası (.pth)
        </label>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            type="text"
            value={modelPath}
            onChange={e => onModelPathChange(e.target.value)}
            placeholder="./models/resnet18_best.pth"
            style={{
              flex: 1, padding: "9px 12px",
              background: "var(--bg-tertiary, #0f172a)",
              border: "1px solid rgba(255,255,255,0.07)",
              borderRadius: 8, color: "var(--text-primary)",
              fontSize: 12, fontFamily: "monospace",
              outline: "none",
            }}
          />
          <button
            onClick={async () => {
              const api = window.electronAPI;
              if (api && api.selectFile) {
                const path = await api.selectFile({ filters: [{ name: "PyTorch Model", extensions: ["pth", "pt"] }] });
                if (path) onModelPathChange(path);
              }
            }}
            style={{
              padding: "9px 14px",
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8, color: "var(--text-secondary)",
              fontSize: 12, cursor: "pointer",
              whiteSpace: "nowrap",
            }}
          >
            Gözat
          </button>
        </div>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------
// Görüntü Yükleyici (Drag & Drop)
// -----------------------------------------------------------------------
function ImageUploader({ imagePath, imagePreview, onImageSelect, onClear }) {
  const fileRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;

    // 1. Electron API'miz var mı kontrol edelim
    const api = window.electronAPI;
    let fullPath = file.name; // Varsayılan olarak sadece isim (Web'de açılırsa diye)

    // 2. Eğer Electron'daysak gerçek yolu webUtils ile alalım
    if (api && api.getFilePath) {
      fullPath = api.getFilePath(file);
    }

    const url = URL.createObjectURL(file);
    
    // 3. Artık gerçek yolu (örneğin: C:\Kullanicilar\Resim.png) başarıyla gönderiyoruz
    onImageSelect(fullPath, url);

    api.logToTerminal("Seçilen Dosyanın Tam Yolu:" + fullPath)
  }, [onImageSelect]);

  const handleDrop = useCallback((e) => {
    e.preventDefault(); 
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  if (imagePreview) {
    return (
      <div style={{ position: "relative", borderRadius: 12, overflow: "hidden", background: "var(--bg-tertiary, #0f172a)" }}>
        <img
          src={imagePreview}
          alt="Seçilen görüntü"
          style={{ width: "100%", height: 220, objectFit: "contain", display: "block" }}
        />
        <button
          onClick={onClear}
          style={{
            position: "absolute", top: 8, right: 8,
            background: "rgba(0,0,0,0.7)", border: "none", borderRadius: "50%",
            width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
            cursor: "pointer", color: "#fff",
          }}
        >
          <X size={14} />
        </button>
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          background: "linear-gradient(transparent, rgba(0,0,0,0.8))",
          padding: "20px 12px 10px", fontSize: 11, color: "rgba(255,255,255,0.6)",
          fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        }}>
          {imagePath}
        </div>
      </div>
    );
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => fileRef.current?.click()}
      style={{
        border: `2px dashed ${dragging ? "#3b82f6" : "rgba(255,255,255,0.1)"}`,
        borderRadius: 12, height: 200,
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        gap: 12, cursor: "pointer",
        background: dragging ? "rgba(59,130,246,0.05)" : "var(--bg-tertiary, #0f172a)",
        transition: "all 0.2s",
      }}
    >
      <div style={{
        width: 48, height: 48, borderRadius: 12,
        background: "rgba(59,130,246,0.15)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <Upload size={22} style={{ color: "#3b82f6" }} />
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 4 }}>
          Görüntü sürükle & bırak
        </div>
        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
          veya tıkla seç · JPG, PNG, BMP, TIFF
        </div>
      </div>
      <input
        ref={fileRef} type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={e => handleFile(e.target.files[0])}
      />
    </div>
  );
}

// -----------------------------------------------------------------------
// Güven Göstergesi
// -----------------------------------------------------------------------
function ConfidenceBar({ score, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div style={{
        flex: 1, height: 6, borderRadius: 999,
        background: "rgba(255,255,255,0.07)",
        overflow: "hidden",
      }}>
        <div style={{
          height: "100%", width: `${(score * 100).toFixed(1)}%`,
          background: color, borderRadius: 999,
          transition: "width 0.6s ease",
          boxShadow: `0 0 8px ${color}66`,
        }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 700, color, minWidth: 46, textAlign: "right" }}>
        %{(score * 100).toFixed(2)}
      </span>
    </div>
  );
}

// -----------------------------------------------------------------------
// Sonuç Paneli
// -----------------------------------------------------------------------
function ResultPanel({ result, modelArch }) {
  const [showAll, setShowAll] = useState(false);
  if (!result) return null;

  const level = getConfidenceLevel(result.confidence);
  const sel = MODELS.find(m => m.id === modelArch) || MODELS[0];

  // Top-5 bar chart için veri
  const chartData = result.top5.map(item => ({
    name: item.class.split(".").slice(-1)[0], // kısa isim
    fullName: item.class,
    score: parseFloat((item.score * 100).toFixed(3)),
  }));

  // Tüm skorlar (sorted, non-zero)
  const allSorted = Object.entries(result.all_scores)
    .sort((a, b) => b[1] - a[1])
    .filter(([, v]) => v > 0.0001);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn 0.4s ease" }}>

      {/* Ana Sonuç Kartı */}
      <div style={{
        padding: "20px 22px",
        background: `linear-gradient(135deg, ${level.color}14, ${sel.color}0a)`,
        border: `1px solid ${level.color}33`,
        borderRadius: 14, position: "relative", overflow: "hidden",
      }}>
        <div style={{
          position: "absolute", top: -30, right: -30,
          width: 100, height: 100, borderRadius: "50%",
          background: `${level.color}0a`,
        }} />
        <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
          <div style={{
            width: 44, height: 44, borderRadius: 11,
            background: `${level.color}20`,
            display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
          }}>
            <Microscope size={20} style={{ color: level.color }} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>
              Tahmin Edilen Bakteri
            </div>
            <div style={{
              fontSize: 18, fontWeight: 700, color: "var(--text-primary)",
              wordBreak: "break-word", lineHeight: 1.3, marginBottom: 10,
              fontFamily: "monospace",
            }}>
              {result.prediction}
            </div>
            <ConfidenceBar score={result.confidence} color={level.color} />
          </div>
        </div>

        {/* Meta bilgiler */}
        <div style={{ display: "flex", gap: 16, marginTop: 16, paddingTop: 14, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Güven</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: level.color }}>{level.label}</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Süre</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-secondary)" }}>{result.inference_time_ms} ms</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Model</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: sel.color }}>{sel.label}</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Cihaz</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-secondary)" }}>{result.device?.toUpperCase()}</div>
          </div>
        </div>
      </div>

      {/* Top-5 Bar Chart */}
      <div className="card">
        <div className="card-header" style={{ paddingBottom: 8 }}>
          <h3 style={{ fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
            <FlaskConical size={14} /> Top-5 Softmax Skorları
          </h3>
        </div>
        <div className="card-body">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 40, top: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} stroke="#475569" tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="name" stroke="#475569" tick={{ fontSize: 11, fontFamily: "monospace" }} width={130} />
              <Tooltip
                contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 }}
                formatter={(v, _, props) => [`%${v.toFixed(3)}`, props.payload.fullName]}
                cursor={{ fill: "rgba(255,255,255,0.03)" }}
              />
              <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={BAR_COLORS[i] || "#64748b"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top-5 Detay Listesi */}
      <div className="card">
        <div className="card-header" style={{ paddingBottom: 8 }}>
          <h3 style={{ fontSize: 13 }}>Detay Sıralaması</h3>
        </div>
        <div className="card-body" style={{ padding: "8px 16px 16px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {result.top5.map((item, i) => (
              <div key={item.class}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{
                      width: 20, height: 20, borderRadius: 5,
                      background: `${BAR_COLORS[i]}22`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 10, fontWeight: 700, color: BAR_COLORS[i],
                    }}>
                      {i + 1}
                    </div>
                    <span style={{ fontSize: 12, fontFamily: "monospace", color: i === 0 ? "var(--text-primary)" : "var(--text-secondary)" }}>
                      {item.class}
                    </span>
                  </div>
                </div>
                <ConfidenceBar score={item.score} color={BAR_COLORS[i] || "#64748b"} />
              </div>
            ))}
          </div>

          {/* Tüm sınıfları göster toggle */}
          <button
            onClick={() => setShowAll(s => !s)}
            style={{
              marginTop: 14, width: "100%", padding: "8px",
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.07)",
              borderRadius: 8, color: "var(--text-muted)", fontSize: 12, cursor: "pointer",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            }}
          >
            <ChevronDown size={14} style={{ transform: showAll ? "rotate(180deg)" : "none", transition: "0.2s" }} />
            {showAll ? "Gizle" : `Tüm 33 sınıfı göster`}
          </button>

          {showAll && (
            <div style={{
              marginTop: 10, maxHeight: 300, overflowY: "auto",
              display: "flex", flexDirection: "column", gap: 6,
            }}>
              {allSorted.map(([cls, score]) => (
                <div key={cls} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{
                    flex: 1, fontSize: 11, fontFamily: "monospace",
                    color: result.prediction === cls ? "#10b981" : "var(--text-muted)",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {cls}
                  </span>
                  <span style={{
                    fontSize: 11, minWidth: 60, textAlign: "right",
                    color: score > 0.01 ? "var(--text-secondary)" : "var(--text-muted)",
                  }}>
                    %{(score * 100).toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------
// HATA GÖSTERICI
// -----------------------------------------------------------------------
function ErrorBox({ message }) {
  return (
    <div style={{
      padding: "14px 16px",
      background: "rgba(239,68,68,0.08)",
      border: "1px solid rgba(239,68,68,0.25)",
      borderRadius: 10, display: "flex", gap: 10, alignItems: "flex-start",
    }}>
      <AlertCircle size={16} style={{ color: "#ef4444", flexShrink: 0, marginTop: 1 }} />
      <div style={{ fontSize: 13, color: "#fca5a5", lineHeight: 1.5 }}>{message}</div>
    </div>
  );
}

// -----------------------------------------------------------------------
// ANA PANEL
// -----------------------------------------------------------------------
export default function InferencePanel() {
  const [selectedArch, setSelectedArch]     = useState("ResNet18");
  const [modelPath, setModelPath]           = useState("");
  const [imagePath, setImagePath]           = useState("");
  const [imagePreview, setImagePreview]     = useState(null);
  const [loading, setLoading]               = useState(false);
  const [result, setResult]                 = useState(null);
  const [error, setError]                   = useState(null);

  const handleImageSelect = useCallback((path, previewUrl) => {
    setImagePath(path);
    setImagePreview(previewUrl);
    setResult(null);
    setError(null);
  }, []);

  const handleClearImage = useCallback(() => {
    setImagePath("");
    setImagePreview(null);
    setResult(null);
    setError(null);
  }, []);

  const handleRunInference = useCallback(async () => {
    if (!imagePath) { setError("Lütfen önce bir görüntü seçin."); return; }
    if (!modelPath) { setError("Lütfen .pth model dosyasının yolunu girin."); return; }

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const api = window.electronAPI;

      if (!api?.runInference) {
        // Geliştirme ortamı için mock
        await new Promise(r => setTimeout(r, 1200));
        setResult({
          prediction:        "Staphylococcus.aureus",
          confidence:        0.8734,
          top5: [
            { class: "Staphylococcus.aureus",         score: 0.8734 },
            { class: "Staphylococcus.epidermidis",    score: 0.0821 },
            { class: "Staphylococcus.saprophiticus",  score: 0.0231 },
            { class: "Micrococcus.spp",               score: 0.0112 },
            { class: "Streptococcus.agalactiae",      score: 0.0043 },
          ],
          all_scores: Object.fromEntries(
            Array.from({ length: 33 }, (_, i) => [
              Object.keys({
                'Acinetobacter.baumanii':0,'Actinomyces.israeli':1,'Bacteroides.fragilis':2,
                'Bifidobacterium.spp':3,'Candida.albicans':4,'Clostridium.perfringens':5,
                'Enterococcus.faecalis':6,'Enterococcus.faecium':7,'Escherichia.coli':8,
                'Fusobacterium':9,'Lactobacillus.casei':10,'Lactobacillus.crispatus':11,
                'Lactobacillus.delbrueckii':12,'Lactobacillus.gasseri':13,'Lactobacillus.jehnsenii':14,
                'Lactobacillus.johnsonii':15,'Lactobacillus.paracasei':16,'Lactobacillus.plantarum':17,
                'Lactobacillus.reuteri':18,'Lactobacillus.rhamnosus':19,'Lactobacillus.salivarius':20,
                'Listeria.monocytogenes':21,'Micrococcus.spp':22,'Neisseria.gonorrhoeae':23,
                'Porfyromonas.gingivalis':24,'Propionibacterium.acnes':25,'Proteus':26,
                'Pseudomonas.aeruginosa':27,'Staphylococcus.aureus':28,'Staphylococcus.epidermidis':29,
                'Staphylococcus.saprophiticus':30,'Streptococcus.agalactiae':31,'Veionella':32
              })[i],
              parseFloat((Math.random() * 0.01).toFixed(6))
            ])
          ),
          inference_time_ms: 38.2,
          device: "cpu",
          model_arch: selectedArch,
        });
        return;
      }

      // Gerçek Electron çağrısı
      const raw = await api.runInference({
        imagePath,
        modelPath,
        arch: selectedArch,
      });

      if (!raw.success) {
        setError(raw.error || "Inference başarısız oldu.");
      } else {
        setResult(raw);
      }
    } catch (err) {
      setError(`Beklenmeyen hata: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [imagePath, modelPath, selectedArch]);

  const canRun = !!imagePath && !!modelPath && !loading;
  const selModel = MODELS.find(m => m.id === selectedArch) || MODELS[0];

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Inference</h2>
        <p>Eğitilmiş modellerle bakteriyel görüntü sınıflandırması</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, alignItems: "start" }}>

        {/* SOL: Giriş Paneli */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Görüntü Yükleme */}
          <div className="card">
            <div className="card-header">
              <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Upload size={15} /> Görüntü Seç
              </h3>
            </div>
            <div className="card-body">
              <ImageUploader
                imagePath={imagePath}
                imagePreview={imagePreview}
                onImageSelect={handleImageSelect}
                onClear={handleClearImage}
              />
            </div>
          </div>

          {/* Model Seçimi */}
          <div className="card">
            <div className="card-header">
              <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Cpu size={15} /> Model Seç
              </h3>
            </div>
            <div className="card-body">
              <ModelSelector
                selected={selectedArch}
                onSelect={setSelectedArch}
                modelPath={modelPath}
                onModelPathChange={setModelPath}
              />
            </div>
          </div>

          {/* Çalıştır Butonu */}
          <button
            onClick={handleRunInference}
            disabled={!canRun}
            style={{
              width: "100%", padding: "14px",
              background: canRun
                ? `linear-gradient(135deg, ${selModel.color}, ${selModel.color}bb)`
                : "rgba(255,255,255,0.06)",
              border: "none", borderRadius: 12,
              color: canRun ? "#fff" : "var(--text-muted)",
              fontSize: 14, fontWeight: 700, cursor: canRun ? "pointer" : "not-allowed",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
              transition: "all 0.2s",
              boxShadow: canRun ? `0 4px 20px ${selModel.color}44` : "none",
            }}
          >
            {loading ? (
              <>
                <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} />
                Analiz ediliyor...
              </>
            ) : (
              <>
                <Microscope size={18} />
                Sınıflandır
              </>
            )}
          </button>

          {error && <ErrorBox message={error} />}
        </div>

        {/* SAĞ: Sonuç Paneli */}
        <div>
          {!result && !loading && (
            <div style={{
              height: 300, display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center", gap: 12,
              color: "var(--text-muted)", textAlign: "center",
            }}>
              <div style={{
                width: 64, height: 64, borderRadius: 16,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.06)",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                <Microscope size={28} style={{ opacity: 0.3 }} />
              </div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>Sonuç bekleniyor</div>
                <div style={{ fontSize: 12 }}>Görüntü ve model seçip "Sınıflandır"a basın</div>
              </div>
            </div>
          )}

          {loading && (
            <div style={{
              height: 300, display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center", gap: 16,
            }}>
              <div style={{
                width: 56, height: 56, borderRadius: "50%",
                border: `3px solid ${selModel.color}33`,
                borderTop: `3px solid ${selModel.color}`,
                animation: "spin 0.8s linear infinite",
              }} />
              <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
                {selModel.label} ile analiz ediliyor...
              </div>
            </div>
          )}

          {result && <ResultPanel result={result} modelArch={selectedArch} />}
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
      `}</style>
    </div>
  );
}