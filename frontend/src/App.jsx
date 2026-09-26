import { useEffect, useMemo, useRef, useState } from "react";

const allowedTypes = ["image/png", "image/jpeg", "image/bmp", "image/webp"];
const fallbackModels = [{ key: "cnn", label: "Shallow CNN", available: false }];

export default function App() {
  const [files, setFiles] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState(fallbackModels);
  const [selectedModel, setSelectedModel] = useState("cnn");
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const fileInputRef = useRef(null);
  const previews = useMemo(() => files.map((file) => URL.createObjectURL(file)), [files]);

  useEffect(() => {
    fetch("/api/models")
      .then((response) => response.json())
      .then((data) => {
        setModels(data.models);
        const preferred = data.models.find((item) => item.key === data.default_model && item.available);
        setSelectedModel((preferred ?? data.models.find((item) => item.available))?.key ?? data.default_model);
        setModelsLoaded(true);
      })
      .catch(() => { setModelsLoaded(true); setError("Could not check the available models. Is the backend running?"); });
  }, []);

  const modelReady = models.some((item) => item.key === selectedModel && item.available);

  useEffect(() => () => previews.forEach(URL.revokeObjectURL), [previews]);

  function chooseFiles(event) {
    const selected = Array.from(event.target.files ?? []);
    const invalid = selected.find((file) => !allowedTypes.includes(file.type) || file.size > 5 * 1024 * 1024);
    if (invalid) {
      setError(`${invalid.name} must be a PNG, JPG, BMP or WebP image under 5 MB.`);
      return;
    }
    setFiles(selected.slice(0, 30));
    setResult(null);
    setError("");
  }

  async function recognise() {
    if (!files.length) return setError("Upload at least one digit image first.");
    setLoading(true);
    setError("");
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    body.append("model", selectedModel);
    try {
      const response = await fetch("/api/predict", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed.");
      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFiles([]);
    setResult(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  return (
    <main>
      <header className="hero">
        <p className="eyebrow">COS30018 · INTELLIGENT SYSTEMS</p>
        <h1>Handwritten Number Recognition</h1>
        <span className={`status ${modelReady ? "ready" : "waiting"}`}>
          {!modelsLoaded ? "Checking models…" : modelReady ? "Model ready" : "Model not trained"}
        </span>
      </header>

      <section className="workspace">
        <article className="panel">
          <h2>Select digit images</h2>
          <p className="hint">Upload one handwritten number image, or several digit images from left to right.</p>
          <label className="dropzone">
            <input ref={fileInputRef} type="file" multiple accept="image/png,image/jpeg,image/bmp,image/webp" onChange={chooseFiles} />
            <strong>Choose image files</strong>
            <span>PNG, JPG, BMP or WebP · maximum 5 MB each</span>
          </label>
          <div className="previews">
            {previews.map((url, index) => (
              <figure key={`${files[index].name}-${index}`}>
                <img src={url} alt={`Uploaded digit ${index + 1}`} />
                <figcaption>{index + 1}</figcaption>
              </figure>
            ))}
          </div>
        </article>

        <article className="panel result-panel">
          <h2>Recognition result</h2>
          <label className="model-selector" htmlFor="model-choice">
            <span>Select model</span>
            <select id="model-choice" value={selectedModel} disabled={loading} onChange={(event) => {
              setSelectedModel(event.target.value);
              setResult(null);
              setError("");
            }}>
              {models.map((item) => (
                <option key={item.key} value={item.key} disabled={!item.available}>
                  {item.label}{item.available ? "" : " (not trained)"}
                </option>
              ))}
            </select>
          </label>
          {!result ? (
            <div className="empty-result">Your prediction will appear here.</div>
          ) : (
            <div className="result">
              <p>Model: {result.model_label}</p>
              <p>Predicted number</p>
              <strong>{result.number}</strong>
              <dl>
                <div><dt>Average confidence</dt><dd>{(result.average_confidence * 100).toFixed(1)}%</dd></div>
                <div><dt>Lowest digit</dt><dd>{(result.lowest_confidence * 100).toFixed(1)}%</dd></div>
              </dl>
              <div className="digit-results">
                {result.predictions.map((item) => (
                  <span key={item.position}>{item.digit}<small>{(item.confidence * 100).toFixed(1)}%</small></span>
                ))}
              </div>
            </div>
          )}
          {error && <p className="error" role="alert">{error}</p>}
          <div className="result-actions">
            <button onClick={recognise} disabled={loading || !files.length || !modelReady}>
              {loading ? "Recognising…" : "Recognise number"}
            </button>
            <button className="secondary-button" onClick={reset} disabled={loading || (!files.length && !result && !error)}>
              Reset
            </button>
          </div>
        </article>
      </section>
    </main>
  );
}
