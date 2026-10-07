import { useState } from "react";
import {
  Search,
  Activity,
  Database,
  Zap,
  AlertTriangle,
  Download,
  Car,
  Clock,
  Layers3,
} from "lucide-react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const scenes = [
  "scene-0061",
  "scene-0103",
  "scene-0553",
  "scene-0655",
  "scene-0757",
  "scene-0796",
  "scene-0916",
  "scene-1077",
  "scene-1094",
  "scene-1100",
];

function App() {
  const [query, setQuery] = useState("night driving");
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);

  const [scene, setScene] = useState("scene-0103");
  const [threshold, setThreshold] = useState(3);
  const [events, setEvents] = useState([]);
  const [mining, setMining] = useState(false);

  async function searchScenes() {
    if (!query.trim()) return;

    setSearching(true);

    try {
      const response = await fetch(
        `${API}/search/text?q=${encodeURIComponent(query)}&k=6`
      );
      const data = await response.json();
      setResults(data.results || []);
    } catch (error) {
      console.error(error);
    } finally {
      setSearching(false);
    }
  }

  async function mineBraking() {
    setMining(true);

    try {
      const response = await fetch(
        `${API}/scenarios/hard-braking/${scene}?threshold=${threshold}`
      );
      const data = await response.json();
      setEvents(data.events || []);
    } catch (error) {
      console.error(error);
      setEvents([]);
    } finally {
      setMining(false);
    }
  }

  function imageUrl(path) {
    return `${API}/images/${path}`;
  }

  async function exportScenario(event) {
    const response = await fetch(
      `${API}/scenarios/export?scene=${scene}&event_index=${event.index}`,
      { method: "POST" }
    );

    const data = await response.json();

    const blob = new Blob(
      [JSON.stringify(data.scenario, null, 2)],
      { type: "application/json" }
    );

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${scene}_${event.index}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Car size={19} />
          </div>
          <div>
            <div className="brand-name">DriveMine</div>
            <div className="brand-subtitle">
              Autonomous Driving Scenario Intelligence
            </div>
          </div>
        </div>

        <div className="api-status">
          <span className="status-dot" />
          API CONNECTED
        </div>
      </header>

      <main>
        <section className="hero">
          <div>
            <div className="eyebrow">NUScenes · SCENARIO MINING</div>
            <h1>Find the moments<br />that matter.</h1>
            <p>
              Search driving logs semantically, detect safety-critical events,
              and export structured scenarios for downstream analysis.
            </p>
          </div>

          <div className="hero-stats">
            <Stat icon={<Database size={17} />} value="404" label="CAM_FRONT frames" />
            <Stat icon={<Layers3 size={17} />} value="512D" label="CLIP embeddings" />
            <Stat icon={<Zap size={17} />} value="1.44×" label="C++ speedup" />
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">
                <Search size={15} />
                SEMANTIC SEARCH
              </div>
              <h2>Search driving scenes</h2>
            </div>
            <span className="tech-label">CLIP + FAISS</span>
          </div>

          <div className="search-row">
            <Search size={18} />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && searchScenes()}
              placeholder="Describe a driving situation..."
            />
            <button onClick={searchScenes} disabled={searching}>
              {searching ? "Searching..." : "Search"}
            </button>
          </div>

          <div className="chips">
            {["night driving", "pedestrian crossing", "rain", "truck ahead"].map(
              (item) => (
                <button
                  key={item}
                  className="chip"
                  onClick={() => {
                    setQuery(item);
                  }}
                >
                  {item}
                </button>
              )
            )}
          </div>

          {results.length > 0 && (
            <div className="results">
              {results.map((result, index) => (
                <div className="result-card" key={`${result.sample_token}-${index}`}>
                  <div className="image-placeholder">
                    <img
                      src={imageUrl(result.cam_path)}
                      alt={`${result.scene} driving scene`}
                    />

                    <div className="image-overlay">
                      <span>{result.scene}</span>
                      <span>{result.score.toFixed(3)}</span>
                    </div>
                  </div>

                  <div className="result-info">
                    <div>
                      <span className="result-rank">0{index + 1}</span>
                      <span className="result-scene">{result.scene}</span>
                    </div>
                    <div className="result-meta">
                      <Clock size={13} />
                      {result.ts}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="panel mining-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">
                <Activity size={15} />
                SCENARIO DETECTION
              </div>
              <h2>Hard braking miner</h2>
            </div>
            <span className="tech-label">C++ / PYBIND11</span>
          </div>

          <div className="controls">
            <div className="control">
              <label>SCENE</label>
              <select value={scene} onChange={(e) => setScene(e.target.value)}>
                {scenes.map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </div>

            <div className="control threshold">
              <label>THRESHOLD · m/s²</label>
              <input
                type="number"
                step="0.5"
                min="0.5"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
              />
            </div>

            <button className="mine-button" onClick={mineBraking}>
              <Activity size={16} />
              {mining ? "Mining..." : "Run detector"}
            </button>
          </div>

          {events.length > 0 ? (
            <div className="events">
              <div className="event-banner">
                <div className="event-icon">
                  <AlertTriangle size={20} />
                </div>
                <div>
                  <strong>{events.length} hard braking event{events.length > 1 ? "s" : ""} detected</strong>
                  <span>{scene} · threshold −{threshold} m/s²</span>
                </div>
              </div>

              {events.map((event) => (
                <div className="event-row" key={event.index}>
                  <div className="event-number">EVENT {String(event.index).padStart(2, "0")}</div>
                  <div className="event-token">{event.sample_token}</div>
                  <div className="event-time">{event.timestamp}</div>
                  <button
                    className="export-button"
                    onClick={() => exportScenario(event)}
                  >
                    <Download size={14} />
                    Export JSON
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <Activity size={20} />
              <span>Run the detector to find braking events in this scene.</span>
            </div>
          )}
        </section>

        <footer>
          <span>DRIVEMINE v0.1</span>
          <span>nuScenes · FastAPI · FAISS · pybind11</span>
        </footer>
      </main>
    </div>
  );
}

function Stat({ icon, value, label }) {
  return (
    <div className="stat">
      <div className="stat-icon">{icon}</div>
      <div>
        <strong>{value}</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

export default App;
