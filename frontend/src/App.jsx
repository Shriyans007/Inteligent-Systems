import { useEffect, useMemo, useRef, useState } from "react";

const allowedTypes = ["image/png", "image/jpeg", "image/bmp", "image/webp"];
const fallbackModels = [{ key: "cnn", label: "Shallow CNN", available: false }];

export default function App() {
  const [source, setSource] = useState("draw");
  const [files, setFiles] = useState([]);
  const [drawn, setDrawn] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState(fallbackModels);
  const [characterModels, setCharacterModels] = useState(fallbackModels);
  const [choice, setChoice] = useState("character:cnn");
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [segmenters, setSegmenters] = useState({ defaults: {}, segmenters: [] });
  const [segmentation, setSegmentation] = useState("");
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const pointerRef = useRef(null);
  const previews = useMemo(() => files.map((file) => URL.createObjectURL(file)), [files]);
  const [family, modelKey] = choice.split(":");
  const modelReady = (family === "character" ? characterModels : models)
    .some((item) => item.key === modelKey && item.available);
  // Until the user picks one, each route uses its own default segmentation method.
  const chosenSegmentation = segmentation || segmenters.defaults[family === "numbers" ? "numbers" : "characters"] || "";
  // Only the digit models report a test accuracy to set the threshold from.
  const selectedAccuracy = family === "numbers" ? models.find((item) => item.key === modelKey)?.accuracy : undefined;

  useEffect(() => {
    Promise.all([fetch("/api/models"), fetch("/api/extension-models")])
      .then(async ([numbers, extensions]) => {
        if (!numbers.ok || !extensions.ok) throw new Error("Model status unavailable.");
        const digit = await numbers.json();
        const text = await extensions.json();
        const characters = text.character_models ?? fallbackModels;
        setModels(digit.models);
        setCharacterModels(characters);
        // Use a 62-class model when available, so the drawing can contain letters and digits.
        const firstCharacter = characters.find((item) => item.key === "cnn" && item.available)
          ?? characters.find((item) => item.available);
        const firstDigit = digit.models.find((item) => item.key === digit.default_model && item.available)
          ?? digit.models.find((item) => item.available);
        setChoice(firstCharacter ? `character:${firstCharacter.key}`
          : firstDigit ? `numbers:${firstDigit.key}` : "character:cnn");
        setModelsLoaded(true);
      })
      .catch(() => { setModelsLoaded(true); setError("Could not check the available models. Is the backend running?"); });
  }, []);
  useEffect(() => {
    // Older backends have no segmentation choice, so a failure just hides the selector.
    fetch("/api/segmenters").then((response) => (response.ok ? response.json() : null))
      .then((data) => { if (data) setSegmenters(data); }).catch(() => {});
  }, []);
  useEffect(() => () => previews.forEach(URL.revokeObjectURL), [previews]);
  // Test accuracy is measured across a dataset; it is not a confidence cutoff
  // for an individual drawing. Keep the user's warning setting independent.
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
  }, [source]);

  function chooseFiles(event) {
    const selected = Array.from(event.target.files ?? []);
    const invalid = selected.find((file) => !allowedTypes.includes(file.type) || file.size > 5 * 1024 * 1024);
    if (invalid) return setError(`${invalid.name} must be a PNG, JPG, BMP or WebP image under 5 MB.`);
    setFiles(selected.slice(0, family === "numbers" ? 30 : 1));
    setResult(null); setError("");
  }
  function changeChoice(event) {
    setChoice(event.target.value);
    setFiles([]); setResult(null); setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
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
    if (canvas) {
      const context = canvas.getContext("2d");
      context.fillStyle = "white";
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
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
      body.append(family === "numbers" ? "files" : "file", blob, "drawing.png");
    } else if (family === "numbers") files.forEach((file) => body.append("files", file));
    else body.append("file", files[0]);
    body.append("model", modelKey);
    if (chosenSegmentation) body.append("segmentation", chosenSegmentation);
    try {
      const response = await fetch(family === "numbers" ? "/api/predict" : "/api/recognise-text", { method: "POST", body });
      const data = await response.json().catch(() => null);
      if (!response.ok) throw new Error(data?.detail || `Prediction failed (API ${response.status}). Check the backend terminal.`);
      if (!data) throw new Error("The API returned an invalid response. Check the backend terminal.");
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
          <h2>{source === "draw" ? "Draw here" : "Upload image"}</h2>
          <p className="hint">{family === "character"
            ? "Draw or upload separate letters and digits from left to right. Spaces and full lines are not supported."
            : "Draw a number or upload a number image. This model recognises digits only."}</p>
          <div className="mode-options" role="group" aria-label="Input method">
            {[["draw", "Draw here"], ["upload", "Upload image"]].map(([key, label]) => (
              <button key={key} className={source === key ? "chosen" : "secondary-button"} disabled={loading}
                onClick={() => { setSource(key); setFiles([]); setResult(null); setError(""); }}>
                {label}
              </button>
            ))}
          </div>
          {source === "upload" ? <>
            <label className="dropzone">
              <input ref={fileInputRef} type="file" multiple={family === "numbers"}
                accept="image/png,image/jpeg,image/bmp,image/webp" onChange={chooseFiles} />
              <strong>Choose image {family === "numbers" ? "files" : "file"}</strong>
              <span>PNG, JPG, BMP or WebP · maximum 5 MB each</span>
            </label>
            <div className="previews">{previews.map((url, index) => (
              <figure key={`${files[index].name}-${index}`}><img src={url} alt={`Uploaded image ${index + 1}`} /><figcaption>{index + 1}</figcaption></figure>
            ))}</div>
          </> : <div className="drawing-area">
            <canvas ref={canvasRef} width="720" height="220" aria-label="Draw handwriting here"
              onPointerDown={startStroke} onPointerMove={moveStroke} onPointerUp={endStroke} onPointerCancel={endStroke} />
            <button className="secondary-button" onClick={clearDrawing} disabled={loading || !drawn}>Clear drawing</button>
          </div>}
        </article>
        <article className="panel result-panel">
          <h2>Recognition result</h2>
          <label className="model-selector" htmlFor="model-choice"><span>Select model</span>
            <select id="model-choice" value={choice} disabled={loading} onChange={changeChoice}>
              <optgroup label="Letters and digits (EMNIST)">
                {characterModels.map((item) => <option key={`character:${item.key}`} value={`character:${item.key}`} disabled={!item.available}>
                  {item.label}{item.available ? "" : " (not trained)"}
                </option>)}
              </optgroup>
              <optgroup label="Digits only (MNIST)">
                {models.map((item) => <option key={`numbers:${item.key}`} value={`numbers:${item.key}`} disabled={!item.available}>
                  {item.label}{item.available ? "" : " (not trained)"}
                </option>)}
              </optgroup>
            </select>
          </label>
          {segmenters.segmenters.length > 0 && <label className="model-selector" htmlFor="segmentation-choice"><span>Segmentation method</span>
            <select id="segmentation-choice" value={chosenSegmentation} disabled={loading}
              onChange={(event) => { setSegmentation(event.target.value); setResult(null); }}>
              {segmenters.segmenters.map((item) => <option key={item.key} value={item.key} disabled={item.available === false}>
                {item.label}{item.available === false ? " (not trained)" : ""}
              </option>)}
            </select>
          </label>}
          {family === "numbers" && <label className="threshold-control" htmlFor="threshold-choice">
            <span>Flag digits below {(threshold * 100).toFixed(0)}% confidence</span>
            <input id="threshold-choice" type="range" min="0" max="100" step="1"
              value={Math.round(threshold * 100)} disabled={loading}
              onChange={(event) => setThreshold(Number(event.target.value) / 100)} />
            {selectedAccuracy != null && <small>MNIST test accuracy: {(selectedAccuracy * 100).toFixed(1)}% (different from confidence on this image)</small>}
          </label>}
          {!modelReady && modelsLoaded && <p className="hint">Train the selected model first, then restart the backend and refresh this page.</p>}
          {!result ? <div className="empty-result">Your prediction will appear here.</div> : "number" in result ? (
            <div className="result"><p>Model: {result.model_label}{result.segmentation_label ? ` · ${result.segmentation_label}` : ""}</p><p>Predicted number</p><strong>{result.number}</strong>
              <dl><div><dt>Average confidence</dt><dd>{(result.average_confidence * 100).toFixed(1)}%</dd></div><div><dt>Lowest digit</dt><dd>{(result.lowest_confidence * 100).toFixed(1)}%</dd></div></dl>
              <div className="digit-results">{result.predictions.map((item) => (
                <span key={item.position} className={item.confidence < threshold ? "uncertain" : undefined}>
                  {item.digit}
                  <small>{(item.confidence * 100).toFixed(1)}%</small>
                  <div className="prob-bars">
                    {item.probabilities.map((p, digit) => (
                      <div key={digit} className={digit === item.digit ? "prob-row winner" : "prob-row"}>
                        <span className="prob-label">{digit}</span>
                        <span className="prob-bar" style={{ width: `${p * 100}%` }} />
                      </div>
                    ))}
                  </div>
                </span>
              ))}</div>
            </div>
          ) : <div className="result"><p>Predicted characters · {result.model_label}{result.segmentation_label ? ` · ${result.segmentation_label}` : ""}</p><strong>{result.text || "(no text detected)"}</strong>
            <p>Average confidence: {(result.confidence * 100).toFixed(1)}%</p>
            <div className="digit-results">{result.predictions?.map((item) => <span key={item.position}>{item.character}<small>{(item.confidence * 100).toFixed(1)}%</small></span>)}</div>
          </div>}
          {error && <p className="error" role="alert">{error}</p>}
          <div className="result-actions">
            <button onClick={recognise} disabled={loading || !modelReady || (source === "upload" ? !files.length : !drawn)}>
              {loading ? "Recognising…" : "Recognise"}
            </button>
            <button className="secondary-button" onClick={reset} disabled={loading || (!files.length && !drawn && !result && !error)}>Reset</button>
          </div>
        </article>
      </section>
    </main>
  );
}
