import { useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [tenantId, setTenantId] = useState(() => crypto.randomUUID());

  const [denialText, setDenialText] = useState("");
  const [eobText, setEobText] = useState("");
  const [policyText, setPolicyText] = useState("");

  const [denialId, setDenialId] = useState("");
  const [eobId, setEobId] = useState("");
  const [policyId, setPolicyId] = useState("");

  const [appeal, setAppeal] = useState(null);
  const [draft, setDraft] = useState("");
  const [message, setMessage] = useState("");

  function newSession() {
    setTenantId(crypto.randomUUID());
    setDenialText("");
    setEobText("");
    setPolicyText("");
    setDenialId("");
    setEobId("");
    setPolicyId("");
    setAppeal(null);
    setDraft("");
    setMessage("Started new ephemeral session.");
  }

  async function saveTextDoc(docType, content, setSavedId) {
    setMessage("");
    const res = await fetch(`${API}/docs`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": tenantId,
      },
      body: JSON.stringify({
        doc_type: docType,
        source_name: `${docType}_input.txt`,
        content,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      setMessage(`Error: ${data.detail || "Could not save document"}`);
      return;
    }
    setSavedId(data.id);
    setMessage(`Saved ${docType} to VM4 (${data.id})`);
  }

  async function savePdfDoc(docType, file, setSavedId) {
    if (!file) return;
    setMessage("");
    const formData = new FormData();
    formData.append("doc_type", docType);
    formData.append("file", file);

    const res = await fetch(`${API}/docs/pdf`, {
      method: "POST",
      headers: { "X-Tenant-ID": tenantId },
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      setMessage(`Error: ${data.detail || "Could not upload PDF"}`);
      return;
    }
    setSavedId(data.id);
    setMessage(`Uploaded ${docType} PDF to VM4 (${data.id})`);
  }

  async function generateAppeal() {
    setMessage("");
    const res = await fetch(`${API}/appeal/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": tenantId,
      },
      body: JSON.stringify({
        denial_id: denialId,
        eob_id: eobId,
        policy_id: policyId,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      setMessage(`Error: ${data.detail || "Could not start appeal"}`);
      return;
    }
    setAppeal(data);
    setDraft(data.draft);
  }

  async function finishApproval(approved) {
    setMessage("");
    const res = await fetch(`${API}/appeal/${appeal.appeal_id}/approve`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": tenantId,
      },
      body: JSON.stringify({ approved, letter: draft }),
    });
    const data = await res.json();
    if (!res.ok) {
      setMessage(`Error: ${data.detail || "Could not update approval"}`);
      return;
    }
    setAppeal(data);
    setMessage(`Appeal status: ${data.status}`);
  }

  return (
    <div style={{ maxWidth: "800px", margin: "20px auto", fontFamily: "sans-serif" }}>
      <h2>ICAE-Cloud Wireframe</h2>
      <p>
        <strong>Session Tenant ID:</strong> {tenantId}{" "}
        <button onClick={newSession}>Reset Session</button>
      </p>
      {message && <p><strong>Status:</strong> {message}</p>}
      <hr />

      <h3>1. Denial Letter</h3>
      <textarea
        rows={4}
        cols={70}
        value={denialText}
        onChange={(e) => setDenialText(e.target.value)}
        placeholder="Paste denial text or JSON..."
      />
      <br />
      <button onClick={() => saveTextDoc("denial", denialText, setDenialId)}>Save Denial Text</button>
      {" | PDF (future): "}
      <input
        type="file"
        accept=".pdf"
        onChange={(e) => savePdfDoc("denial", e.target.files[0], setDenialId)}
      />
      {denialId && <p>Saved ID: {denialId}</p>}

      <h3>2. EOB Document</h3>
      <textarea
        rows={4}
        cols={70}
        value={eobText}
        onChange={(e) => setEobText(e.target.value)}
        placeholder="Paste EOB text or JSON..."
      />
      <br />
      <button onClick={() => saveTextDoc("eob", eobText, setEobId)}>Save EOB Text</button>
      {" | PDF (future): "}
      <input
        type="file"
        accept=".pdf"
        onChange={(e) => savePdfDoc("eob", e.target.files[0], setEobId)}
      />
      {eobId && <p>Saved ID: {eobId}</p>}

      <h3>3. Policy Document</h3>
      <textarea
        rows={4}
        cols={70}
        value={policyText}
        onChange={(e) => setPolicyText(e.target.value)}
        placeholder="Paste policy text or JSON..."
      />
      <br />
      <button onClick={() => saveTextDoc("policy", policyText, setPolicyId)}>Save Policy Text</button>
      {" | PDF (future): "}
      <input
        type="file"
        accept=".pdf"
        onChange={(e) => savePdfDoc("policy", e.target.files[0], setPolicyId)}
      />
      {policyId && <p>Saved ID: {policyId}</p>}

      <hr />
      <h3>4. Generate Appeal</h3>
      <button onClick={generateAppeal} disabled={!denialId || !eobId || !policyId}>
        Start Appeal Workflow
      </button>

{appeal && (
        <div style={{ marginTop: "16px" }}>
          <h4>Appeal Status: {appeal.status}</h4>
          <p>
            <strong>Cited Clauses:</strong>{" "}
            {appeal.cited_clauses && appeal.cited_clauses.length > 0
              ? appeal.cited_clauses.join(", ")
              : "None yet (placeholder)"}
          </p>
          <p>
            <strong>Critique Status:</strong>{" "}
            {appeal.critique_passed === false ? "Failed / Needs Review" : "Passed"}
            {appeal.critique_notes ? ` — ${appeal.critique_notes}` : ""}
          </p>
          <textarea
            rows={8}
            cols={70}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <br />
          <button
            onClick={() => finishApproval(true)}
            disabled={appeal.critique_passed === false}
          >
            Approve
          </button>{" "}
          <button onClick={() => finishApproval(false)}>Reject</button>
        </div>
      )}
    </div>
  );
}