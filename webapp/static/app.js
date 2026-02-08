const { useMemo, useState, useEffect, memo, useCallback, useRef } = React;

// Hoist static config
const DEFAULT_JSON = `[
  {
    "role": "user",
    "content": [
      {"type": "image", "image_url": "https://free-images.com/md/95e5/road_sign_asphalt_road.jpg"},
      {"type": "text", "text": "Describe this image."}
    ]
  }
]`;

const ROLES = {
  user: { label: 'User', color: 'bg-blue-50 text-blue-700 border-blue-100', icon: 'user' },
  assistant: { label: 'Assistant', color: 'bg-purple-50 text-purple-700 border-purple-100', icon: 'bot' },
  system: { label: 'System', color: 'bg-slate-50 text-slate-600 border-slate-200', icon: 'settings' }
};

// Utils
const isLikelyUrl = (val) => typeof val === "string" && (val.startsWith("http") || val.startsWith("data:"));
const mediaSrc = (val) => (!val ? "" : isLikelyUrl(val) ? val : `/api/media?path=${encodeURIComponent(val)}`);

const normalizeMessages = (raw) => {
  if (!Array.isArray(raw)) throw new Error("JSON must be a list of chat messages.");
  return raw.map((msg) => ({
    role: msg.role || "user",
    content: Array.isArray(msg.content) ? msg.content : (typeof msg.content === "string" ? [{ type: "text", text: msg.content }] : [])
  }));
};

const detectMode = (messages) => {
  if (!messages.length) return "";
  return messages[messages.length - 1].role === "user" ? "generate" : "encode";
};

// Components
const Icon = memo(({ name, size = 16, className = "" }) => {
  useEffect(() => {
    try {
      if (window.lucide) window.lucide.createIcons();
    } catch (e) {
      console.warn("Lucide failed to create icons", e);
    }
  }, [name]);
  
  return React.createElement("i", { 
    "data-lucide": name, 
    style: { width: size, height: size }, 
    className: `inline-block ${className}` 
  });
});

const Badge = memo(({ children, active }) => (
  React.createElement("span", { 
    className: `px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider transition-colors ${
      active ? "bg-indigo-100 text-indigo-700 border border-indigo-200" : "bg-slate-100 text-slate-500 border border-slate-200"
    }`
  }, children)
));

const ChatTurn = memo(({ msg }) => {
  const roleConfig = ROLES[msg.role] || ROLES.user;
  return React.createElement("div", { className: `flex gap-3 p-4 rounded-xl border ${roleConfig.color} animate-in` },
    React.createElement("div", { className: "flex-shrink-0 mt-1" }, 
      React.createElement(Icon, { name: roleConfig.icon, size: 18 })
    ),
    React.createElement("div", { className: "flex-grow space-y-3 min-w-0" },
      msg.content.map((part, i) => {
        if (part.type === "text") {
          return React.createElement("p", { key: i, className: "text-sm leading-relaxed break-words" }, part.text);
        }
        if (part.type === "image" || part.type === "image_url") {
          return React.createElement("div", { key: i, className: "relative group w-32 aspect-square rounded-lg overflow-hidden border border-black/5 bg-white" },
            React.createElement("img", { 
              src: mediaSrc(part.image || part.image_url), 
              className: "w-full h-full object-cover transition-transform group-hover:scale-105",
              onError: (e) => e.target.src = "https://via.placeholder.com/150?text=Error"
            })
          );
        }
        if (part.type === "video") {
          return React.createElement("div", { key: i, className: "flex items-center gap-2 p-2 rounded bg-black/5 text-[10px] font-mono" },
            React.createElement(Icon, { name: "video", size: 12 }),
            React.createElement("span", { className: "truncate" }, part.video)
          );
        }
        return null;
      })
    )
  );
});

const ControlInput = memo(({ label, value, onChange, min, max, step, type = "number", className = "" }) => (
  React.createElement("div", { className: `space-y-1 ${className}` },
    React.createElement("label", { className: "text-[9px] font-bold text-slate-400 uppercase tracking-widest block" }, label),
    React.createElement("input", {
      type, min, max, step, value,
      onChange: (e) => onChange(e.target.value),
      className: "w-full px-2 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-xs focus:ring-2 focus:ring-indigo-500/10 focus:border-indigo-500 outline-none transition-all font-medium"
    })
  )
));

const ControlToggle = memo(({ label, value, onChange, className = "" }) => (
  React.createElement("div", { className: `flex items-center justify-between gap-2 ${className}` },
    React.createElement("label", { className: "text-[9px] font-bold text-slate-400 uppercase tracking-widest" }, label),
    React.createElement("button", {
      onClick: () => onChange(!value),
      className: `relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none ${value ? 'bg-indigo-600' : 'bg-slate-200'}`
    },
      React.createElement("span", {
        className: `inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${value ? 'translate-x-5' : 'translate-x-1'}`
      })
    )
  )
));

function App() {
  const [jsonInput, setJsonInput] = useState(DEFAULT_JSON);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("idle");
  const [html, setHtml] = useState("");
  const [generated, setGenerated] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef(null);

  const [config, setConfig] = useState({
    max_new_tokens: 128,
    temperature: 1.0,
    top_p: 1.0,
    top_k: 3
  });

  const [visConfig, setVisConfig] = useState({
    apply_filter: true,
    apply_eci: true,
    kernel_size: 3
  });

  const parsed = useMemo(() => {
    try {
      const raw = JSON.parse(jsonInput);
      const messages = normalizeMessages(raw);
      setError("");
      return messages;
    } catch (err) {
      setError(err.message || "Invalid JSON.");
      return [];
    }
  }, [jsonInput]);

  const mode = useMemo(() => detectMode(parsed), [parsed]);

  const runTAM = useCallback(async () => {
    if (!parsed.length || error) return;
    setStatus("running");
    setGenerated("");
    setHtml("");
    
    try {
      const res = await fetch("/api/visualize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          messages: parsed, 
          generation: config,
          visualization: visConfig
        })
      });
      
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Request failed.");
      }
      
      const data = await res.json();
      if (data.generated_text) setGenerated(data.generated_text);
      setHtml(data.html || "");
      setStatus("ready");
    } catch (err) {
      setStatus("error");
      setGenerated(err.message);
    }
  }, [parsed, error, config, visConfig]);

  const updateConfig = (key, val) => setConfig(prev => ({ ...prev, [key]: val }));
  const updateVisConfig = (key, val) => setVisConfig(prev => ({ ...prev, [key]: val }));

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      const text = await file.text();
      setJsonInput(text);
    }
  };

  return React.createElement("div", { className: "h-screen flex flex-col bg-slate-50 overflow-hidden" },
    // Header
    React.createElement("header", { className: "h-14 flex-shrink-0 flex items-center justify-between px-6 bg-white border-b border-slate-200 z-10" },
      React.createElement("div", { className: "flex items-center gap-3" },
        React.createElement("img", { 
          src: "/static/favicon.ico", 
          className: "w-7 h-7 rounded-lg shadow-lg shadow-indigo-100",
          alt: "TAM Logo"
        }),
        React.createElement("div", null,
          React.createElement("h1", { className: "text-sm font-bold text-slate-900 leading-none" }, "TAM Visualizer"),
          React.createElement("p", { className: "text-[9px] text-slate-400 font-bold uppercase tracking-wider mt-1" }, "VLM Interpretability Studio")
        )
      ),
      React.createElement("div", { className: "flex items-center gap-4" },
        status === "running" && React.createElement("div", { className: "flex items-center gap-2 text-xs text-slate-400 font-semibold animate-pulse" },
          React.createElement("div", { className: "w-2.5 h-2.5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" }),
          "Analyzing..."
        ),
        React.createElement("button", {
          onClick: runTAM,
          disabled: status === "running" || !!error || !parsed.length,
          className: "px-4 py-1.5 bg-slate-900 text-white text-xs font-bold rounded-lg hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md shadow-slate-200 active:scale-95"
        }, 
          React.createElement(Icon, { name: "play", size: 12 }),
          status === "running" ? "Processing..." : "Run Analysis"
        )
      )
    ),

    // Main Content
    React.createElement("main", { className: "flex-grow flex overflow-hidden" },
      // Left Panel
      React.createElement("aside", { className: "w-[400px] flex-shrink-0 border-r border-slate-200 bg-white flex flex-col" },
        React.createElement("div", { className: "flex-grow overflow-y-auto p-5 space-y-6" },
          // JSON Editor
          React.createElement("section", { className: "space-y-2" },
            React.createElement("div", { className: "flex items-center justify-between" },
              React.createElement("h3", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest" }, "Input Messages"),
              error && React.createElement("span", { className: "text-[9px] font-bold text-rose-500 bg-rose-50 px-1.5 py-0.5 rounded border border-rose-100" }, "Syntax Error")
            ),
            React.createElement("div", { 
              className: `relative rounded-xl transition-all ${isDragging ? 'ring-4 ring-indigo-500/20' : ''}`,
              onDragOver: handleDragOver,
              onDragLeave: handleDragLeave,
              onDrop: handleDrop
            },
              React.createElement("textarea", {
                ref: textareaRef,
                value: jsonInput,
                onChange: (e) => setJsonInput(e.target.value),
                placeholder: "Paste JSON or drag a file here...",
                className: `w-full h-44 p-3 font-mono text-[11px] bg-slate-900 text-slate-300 rounded-xl outline-none focus:ring-4 transition-all resize-none ${
                  error ? "focus:ring-rose-500/10 ring-1 ring-rose-500/30" : "focus:ring-indigo-500/10 ring-1 ring-slate-800"
                } ${isDragging ? 'opacity-50' : ''}`
              }),
              isDragging && React.createElement("div", { className: "absolute inset-0 flex items-center justify-center bg-indigo-500/10 rounded-xl border-2 border-dashed border-indigo-500 pointer-events-none" },
                React.createElement("div", { className: "bg-white px-3 py-1.5 rounded-lg shadow-sm text-[10px] font-bold text-indigo-600" }, "Drop to import file")
              )
            )
          ),

          // Controls
          React.createElement("section", { className: "grid grid-cols-2 gap-x-6 gap-y-4" },
            React.createElement("div", { className: "space-y-3" },
              React.createElement("h3", { className: "text-[9px] font-bold text-slate-400 uppercase tracking-widest" }, "Generation"),
              React.createElement("div", { className: "grid grid-cols-2 gap-2" },
                React.createElement(ControlInput, { label: "Tokens", value: config.max_new_tokens, onChange: (v) => updateConfig("max_new_tokens", Number(v)) }),
                React.createElement(ControlInput, { label: "Temp", step: "0.1", value: config.temperature, onChange: (v) => updateConfig("temperature", Number(v)) }),
                React.createElement(ControlInput, { label: "Top P", step: "0.05", value: config.top_p, onChange: (v) => updateConfig("top_p", Number(v)) }),
                React.createElement(ControlInput, { label: "Top K", value: config.top_k, onChange: (v) => updateConfig("top_k", Number(v)) })
              )
            ),
            React.createElement("div", { className: "space-y-3" },
              React.createElement("h3", { className: "text-[9px] font-bold text-slate-400 uppercase tracking-widest" }, "TAM Config"),
              React.createElement("div", { className: "space-y-2" },
                React.createElement(ControlToggle, { label: "Filter", value: visConfig.apply_filter, onChange: (v) => updateVisConfig("apply_filter", v) }),
                React.createElement(ControlToggle, { label: "ECI", value: visConfig.apply_eci, onChange: (v) => updateVisConfig("apply_eci", v) }),
                React.createElement(ControlInput, { label: "Kernel", value: visConfig.kernel_size, onChange: (v) => updateVisConfig("kernel_size", Number(v)) })
              )
            )
          ),

          // Preview
          React.createElement("section", { className: "space-y-3 pb-4" },
            React.createElement("div", { className: "flex items-center justify-between border-b border-slate-100 pb-2" },
              React.createElement("h3", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest" }, "Chat Preview"),
              React.createElement(Badge, { active: !!mode }, mode ? `Mode: ${mode}` : "Empty")
            ),
            React.createElement("div", { className: "space-y-3" },
              parsed.length > 0 
                ? parsed.map((msg, i) => React.createElement(ChatTurn, { key: i, msg }))
                : React.createElement("div", { className: "py-10 text-center rounded-xl border border-dashed border-slate-200" },
                    React.createElement("p", { className: "text-[11px] text-slate-400 font-medium" }, "Enter valid JSON to preview")
                  )
            )
          )
        )
      ),

      // Right Panel
      React.createElement("section", { className: "flex-grow flex flex-col p-4 overflow-hidden" },
        React.createElement("div", { className: "flex-grow flex flex-col bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden" },
          // Toolbar
          React.createElement("div", { className: "h-10 px-4 flex items-center justify-between border-b border-slate-100 bg-white flex-shrink-0" },
            React.createElement("div", { className: "flex items-center gap-2" },
              React.createElement("div", { className: `w-1.5 h-1.5 rounded-full ${status === 'ready' ? 'bg-emerald-500' : status === 'error' ? 'bg-rose-500' : 'bg-slate-300'}` }),
              React.createElement("span", { className: "text-[10px] font-bold text-slate-500 uppercase tracking-wider" }, 
                status === 'idle' ? 'Awaiting Data' : 
                status === 'running' ? 'Processing...' :
                status === 'ready' ? 'Interactive Map Ready' : 'Analysis Failed'
              )
            ),
            (status === 'ready' || status === 'error') && React.createElement("div", { 
              className: `max-w-[60%] truncate text-[10px] font-bold px-3 py-1 rounded-full border ${
                status === 'ready' ? 'text-emerald-600 bg-emerald-50 border-emerald-100' : 'text-rose-600 bg-rose-50 border-rose-100'
              }` 
            },
              status === 'ready' ? (generated ? `Generated: ${generated.length} chars` : "Process Complete") :
              `Error: ${generated}`
            )
          ),
          
          // Iframe
          React.createElement("div", { className: "flex-grow relative bg-slate-100" },
            !html && React.createElement("div", { className: "absolute inset-0 flex items-center justify-center flex-col gap-3" },
              React.createElement("div", { className: "w-12 h-12 rounded-2xl bg-white shadow-sm flex items-center justify-center text-slate-300 border border-slate-200" },
                React.createElement(Icon, { name: "map", size: 24 })
              ),
              React.createElement("div", { className: "text-center space-y-1" },
                React.createElement("p", { className: "text-xs font-bold text-slate-400" }, 
                  status === 'running' ? "Engine Processing" : "Visualizer Inactive"
                ),
                React.createElement("p", { className: "text-[10px] text-slate-400" },
                  status === 'running' ? "TAM scores are being calculated..." : "Click 'Run Analysis' to generate the interactive map"
                )
              )
            ),
            html && React.createElement("iframe", {
              srcDoc: html,
              className: "w-full h-full border-none bg-white",
              title: "TAM Output"
            })
          )
        )
      )
    )
  );
}

// Safer initialization
function init() {
  const container = document.getElementById("root");
  if (container) {
    const root = ReactDOM.createRoot(container);
    root.render(React.createElement(App));
    console.log("TAM App Initialized");
  } else {
    console.error("Root element not found");
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
