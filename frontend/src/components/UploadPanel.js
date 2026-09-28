// UploadPanel.js
import React, { useState, useRef, useCallback } from 'react';

export function UploadPanel({ API, onPrediction }) {
  const inputRef  = useRef(null);
  const [b64,     setB64]     = useState(null);
  const [annB64,  setAnnB64]  = useState(null);
  const [drag,    setDrag]    = useState(false);
  const [loading, setLoading] = useState(false);

  const loadFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const r = new FileReader();
    r.onload = e => { setB64(e.target.result); setAnnB64(null); };
    r.readAsDataURL(file);
  }, []);

  const predict = useCallback(async () => {
    if (!b64) return;
    setLoading(true);
    try {
      const r = await fetch(`${API}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64, source: 'upload' }),
      });
      const d = await r.json();
      onPrediction(d);
      if (d.drawn_image) setAnnB64(d.drawn_image);
    } catch (e) {
      onPrediction({ error: String(e) });
    } finally { setLoading(false); }
  }, [API, b64, onPrediction]);

  const clear = useCallback(() => {
    setB64(null); setAnnB64(null);
    if (inputRef.current) inputRef.current.value = '';
    onPrediction(null);
  }, [onPrediction]);

  const s = {
    wrap: { background: '#fff', border: '1px solid var(--border)',
            borderRadius: 'var(--radius)', overflow: 'hidden',
            boxShadow: 'var(--shadow)' },
    zone: {
      position: 'relative', aspectRatio: '4/3',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: drag ? 'var(--saf-lt)' : b64 ? '#f9f9f9' : 'var(--warm)',
      border: drag ? '2px dashed var(--saffron)' : '2px dashed var(--border)',
      margin: 14, borderRadius: 10,
      cursor: b64 ? 'default' : 'pointer',
      overflow: 'hidden', transition: 'all .2s',
    },
    ph: { textAlign: 'center', padding: 32 },
    controls: { display: 'flex', gap: 10, padding: '0 14px 14px' },
    btn: (p, dis) => ({
      flex: p ? 2 : 1, padding: '11px 0',
      background: dis ? '#ccc' : p ? 'var(--saffron)' : '#fff',
      border: p ? 'none' : '1px solid var(--border)',
      color: p ? '#fff' : 'var(--text2)',
      borderRadius: 8, fontSize: 14, fontWeight: 600,
      cursor: dis ? 'not-allowed' : 'pointer', fontFamily: 'Inter',
    }),
  };

  return (
    <div style={s.wrap}>
      <div style={s.zone}
        onDragOver={e => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); loadFile(e.dataTransfer.files[0]); }}
        onClick={() => !b64 && inputRef.current?.click()}
      >
        {!b64 ? (
          <div style={s.ph}>
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none"
                 stroke="var(--text3)" strokeWidth="1.5" style={{ marginBottom: 12 }}>
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <div style={{ fontWeight: 600, fontSize: 15, color: 'var(--text)',
                          marginBottom: 6 }}>Drop image here</div>
            <div style={{ fontSize: 13, color: 'var(--text3)', lineHeight: 1.6 }}>
              or click to browse<br/>JPG, PNG, WebP, BMP
            </div>
            <div style={{ marginTop: 14, fontSize: 12, color: 'var(--saffron)',
                          background: 'var(--saf-lt)', padding: '6px 14px',
                          borderRadius: 20, display: 'inline-block' }}>
              Plain background recommended
            </div>
          </div>
        ) : (
          <>
            <img src={b64} alt="preview"
                 style={{ position:'absolute', inset:0, width:'100%',
                          height:'100%', objectFit:'contain' }}/>
            {annB64 && (
              <img src={annB64} alt="annotated"
                   style={{ position:'absolute', inset:0, width:'100%',
                            height:'100%', objectFit:'contain', zIndex:2 }}/>
            )}
          </>
        )}
      </div>
      <input ref={inputRef} type="file" accept="image/*"
             style={{ display:'none' }}
             onChange={e => loadFile(e.target.files[0])}/>
      <div style={s.controls}>
        <button style={s.btn(true, !b64 || loading)}
                disabled={!b64 || loading} onClick={predict}>
          {loading ? 'Recognising…' : 'Recognise Mudra'}
        </button>
        <button style={s.btn(false, false)} onClick={clear}>Clear</button>
      </div>
    </div>
  );
}
