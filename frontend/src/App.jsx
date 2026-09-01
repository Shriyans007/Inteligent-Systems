import { useEffect, useMemo, useState } from "react";

const allowedTypes = ["image/png", "image/jpeg", "image/bmp", "image/webp"];

export default function App() {
  const [files, setFiles] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [modelReady, setModelReady] = useState(null);
  const previews = useMemo(() => files.map((file) => URL.createObjectURL(file)), [files]);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then((data) => setModelReady(data.model_ready))
      .catch(() => setModelReady(false));
  }, []);

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

  return (
    <main>
      <header className="hero">
        <p className="eyebrow">COS30018 · INTELLIGENT SYSTEMS</p>
        <h1>Handwritten Number Recognition</h1>
        <p>Upload digits, run the trained CNN and inspect confidence before accepting a prediction.</p>
        <span className={`status ${modelReady ? "ready" : "waiting"}`}>
          {modelReady === null ? "Checking model…" : modelReady ? "Model ready" : "Model not trained"}
        </span>
      </header>

      <section className="workspace">
        <article className="panel">
          <div className="step">01</div>
          <h2>Select digit images</h2>
          <p className="hint">For a multi-digit number, upload pre-segmented images from left to right.</p>
          <label className="dropzone">
            <input type="file" multiple accept="image/png,image/jpeg,image/bmp,image/webp" onChange={chooseFiles} />
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
          <div className="step">02</div>
          <h2>Recognition result</h2>
          {!result ? (
            <div className="empty-result">Your prediction will appear here.</div>
          ) : (
            <div className="result">
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
          <button onClick={recognise} disabled={loading || !files.length}>
            {loading ? "Recognising…" : "Recognise number"}
          </button>
        </article>
      </section>
    </main>
  );
}
