import { useEffect, useMemo, useRef, useState } from "react";

const allowedTypes = ["image/png", "image/jpeg", "image/bmp", "image/webp"];
const fallbackModels = [{ key: "cnn", label: "Shallow CNN", available: false }];

export default function App() {
  const [mode, setMode] = useState("numbers");
  const [source, setSource] = useState("upload");
  const [files, setFiles] = useState([]);
  const [drawn, setDrawn] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState(fallbackModels);
  const [extensionModels, setExtensionModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("cnn");
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const pointerRef = useRef(null);
  const previews = useMemo(() => files.map((file) => URL.createObjectURL(file)), [files]);

  useEffect(() => {
    Promise.all([fetch("/api/models"), fetch("/api/extension-models")])
      .then(async ([numbers, extensions]) => {
        if (!numbers.ok || !extensions.ok) throw new Error("Model status unavailable.");
        const digit = await numbers.json();
        const text = await extensions.json();
        setModels(digit.models);
        setExtensionModels(text.models);
        const preferred = digit.models.find((item) => item.key === digit.default_model && item.available);
        setSelectedModel((preferred ?? digit.models.find((item) => item.available))?.key ?? digit.default_model);
        setModelsLoaded(true);
      })
      .catch(() => { setModelsLoaded(true); setError("Could not check the available models. Is the backend running?"); });
  }, []);
  useEffect(() => () => previews.forEach(URL.revokeObjectURL), [previews]);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "black";
    context.lineWidth = 9;
    context.lineCap = "round";
    context.lineJoin = "round";
    setDrawn(false);
  }, [mode, source]);

  const modelReady = mode === "numbers"
    ? models.some((item) => item.key === selectedModel && item.available)
    : extensionModels.some((item) => item.key === mode && item.available);

  function changeMode(next) {
    setMode(next); setFiles([]); setResult(null); setError(""); setDrawn(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }
  function chooseFiles(event) {
    const selected = Array.from(event.target.files ?? []);
    const invalid = selected.find((file) => !allowedTypes.includes(file.type) || file.size > 5 * 1024 * 1024);
    if (invalid) return setError(`${invalid.name} must be a PNG, JPG, BMP or WebP image under 5 MB.`);
    if (mode !== "numbers" && selected.length > 1) return setError("Select one image for this mode.");
    setFiles(selected.slice(0, mode === "numbers" ? 30 : 1)); setResult(null); setError("");
  }
  function point(event) {
    const bounds = canvasRef.current.getBoundingClientRect();
    return [(event.clientX - bounds.left) * canvasRef.current.width / bounds.width,
      (event.clientY - bounds.top) * canvasRef.current.height / bounds.height];
  }
  function startStroke(event) {
    const canvas = canvasRef.current;
    canvas.setPointerCapture(event.pointerId);
    const [x, y] = point(event);
    const context = canvas.getContext("2d");
    context.beginPath(); context.moveTo(x, y); context.lineTo(x, y); context.stroke();
    pointerRef.current = event.pointerId;
    setDrawn(true); setResult(null); setError("");
  }
  function moveStroke(event) {
    if (pointerRef.current !== event.pointerId) return;
    const [x, y] = point(event);
    const context = canvasRef.current.getContext("2d");
    context.lineTo(x, y); context.stroke();
  }
  function endStroke(event) {
    if (pointerRef.current === event.pointerId) pointerRef.current = null;
  }
  function clearDrawing() {
    const canvas = canvasRef.current;
    if (canvas) { const context = canvas.getContext("2d"); context.fillStyle = "white"; context.fillRect(0, 0, canvas.width, canvas.height); }
    setDrawn(false); setResult(null); setError("");
  }
  async function recognise() {
    if (source === "upload" && !files.length) return setError("Upload an image first.");
    if (source === "draw" && !drawn) return setError("Draw something first.");
    setLoading(true); setError(""); setResult(null);
    const body = new FormData();
    if (source === "draw") {
      const blob = await new Promise((resolve) => canvasRef.current.toBlob(resolve, "image/png"));
      if (!blob) { setLoading(false); return setError("Could not read the drawing."); }
      if (mode === "numbers") body.append("files", blob, "drawing.png");
      else body.append("file", blob, "drawing.png");
    } else if (mode === "numbers") files.forEach((file) => body.append("files", file));
    else body.append("file", files[0]);
    if (mode === "numbers") body.append("model", selectedModel);
    else body.append("mode", mode);
    try {
      const response = await fetch(mode === "numbers" ? "/api/predict" : "/api/recognise-text", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed.");
      setResult(data);
    } catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  }
  function reset() {
    setFiles([]); setResult(null); setError(""); clearDrawing();
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  return (
    <main>
      <header className="hero">
        <p className="eyebrow">COS30018 · INTELLIGENT SYSTEMS</p>
        <h1>Handwritten Recognition</h1>
        <span className={`status ${modelReady ? "ready" : "waiting"}`}>
          {!modelsLoaded ? "Checking models…" : modelReady ? "Model ready" : "Train the model first"}
        </span>
      </header>
      <section className="workspace">
        <article className="panel">
          <h2>What would you like to recognise?</h2>
          <div className="mode-options" role="group" aria-label="Recognition mode">
            {[["numbers", "Numbers"], ["character", "Character"], ["word", "Word"]].map(([key, label]) => (
              <button key={key} className={mode === key ? "chosen" : "secondary-button"} onClick={() => changeMode(key)} disabled={loading}>{label}</button>
            ))}
          </div>
          <p className="hint">{mode === "numbers" ? "Upload a handwritten number or several digit images from left to right." : mode === "character" ? "One letter or digit per image." : "One handwritten word per image. Spaces and full lines are not supported."}</p>
          <div className="mode-options" role="group" aria-label="Input method">
            {[["upload", "Upload image"], ["draw", "Draw here"]].map(([key, label]) => (
              <button key={key} className={source === key ? "chosen" : "secondary-button"} disabled={loading} onClick={() => { setSource(key); setResult(null); setError(""); }}>{label}</button>
            ))}
          </div>
          {source === "upload" ? <>
            <label className="dropzone">
              <input ref={fileInputRef} type="file" multiple={mode === "numbers"} accept="image/png,image/jpeg,image/bmp,image/webp" onChange={chooseFiles} />
              <strong>Choose image {mode === "numbers" ? "files" : "file"}</strong>
              <span>PNG, JPG, BMP or WebP · maximum 5 MB each</span>
            </label>
            <div className="previews">{previews.map((url, index) => (
              <figure key={`${files[index].name}-${index}`}><img src={url} alt={`Uploaded image ${index + 1}`} /><figcaption>{index + 1}</figcaption></figure>
            ))}</div>
          </> : <div className="drawing-area">
            <canvas ref={canvasRef} width="720" height="220" aria-label="Draw handwriting here" onPointerDown={startStroke} onPointerMove={moveStroke} onPointerUp={endStroke} onPointerCancel={endStroke} />
            <button className="secondary-button" onClick={clearDrawing} disabled={loading || !drawn}>Clear drawing</button>
          </div>}
        </article>
        <article className="panel result-panel">
          <h2>Recognition result</h2>
          {mode === "numbers" && <label className="model-selector" htmlFor="model-choice"><span>Select model</span>
            <select id="model-choice" value={selectedModel} disabled={loading} onChange={(event) => { setSelectedModel(event.target.value); setResult(null); setError(""); }}>
              {models.map((item) => <option key={item.key} value={item.key} disabled={!item.available}>{item.label}{item.available ? "" : " (not trained)"}</option>)}
            </select></label>}
          {!modelReady && modelsLoaded && <p className="hint">Train the {mode === "numbers" ? "selected number" : mode} model first, then restart the backend and refresh this page.</p>}
          {!result ? <div className="empty-result">Your prediction will appear here.</div> : mode === "numbers" ? (
            <div className="result"><p>Model: {result.model_label}</p><p>Predicted number</p><strong>{result.number}</strong>
              <dl><div><dt>Average confidence</dt><dd>{(result.average_confidence * 100).toFixed(1)}%</dd></div><div><dt>Lowest digit</dt><dd>{(result.lowest_confidence * 100).toFixed(1)}%</dd></div></dl>
              <div className="digit-results">{result.predictions.map((item) => <span key={item.position}>{item.digit}<small>{(item.confidence * 100).toFixed(1)}%</small></span>)}</div>
            </div>
          ) : <div className="result"><p>Predicted {mode}</p><strong>{result.text || "(no text detected)"}</strong>{mode === "character" && <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>}</div>}
          {error && <p className="error" role="alert">{error}</p>}
          <div className="result-actions">
            <button onClick={recognise} disabled={loading || !modelReady || (source === "upload" ? !files.length : !drawn)}>{loading ? "Recognising…" : `Recognise ${mode === "numbers" ? "number" : mode}`}</button>
            <button className="secondary-button" onClick={reset} disabled={loading || (!files.length && !drawn && !result && !error)}>Reset</button>
          </div>
        </article>
      </section>
    </main>
  );
}
