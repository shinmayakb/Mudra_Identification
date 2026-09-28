import React, { useState, useEffect, useCallback, useRef } from 'react';

export default function TrainModal({ API, onClose, onDone }) {
  const [st,   setSt]   = useState({ progress:0, message:'Starting…',
                                      accuracy:0, phase:'', running:true });
  const [done, setDone] = useState(false);
  const pollRef = useRef(null);

  const poll = useCallback(async () => {
    try {
      const r = await fetch(`${API}/train/status`);
      const d = await r.json();
      setSt(d);
      if (!d.running && d.progress >= 100) {
        clearInterval(pollRef.current);
        setDone(true);
      }
    } catch { /* ignore */ }
  }, [API]);

  useEffect(() => {
    fetch(`${API}/train`, { method:'POST' }).catch(() => {});
    poll();
    pollRef.current = setInterval(poll, 1500);
    return () => clearInterval(pollRef.current);
  }, [API, poll]);

  const overlay = {
    position:'fixed', inset:0, zIndex:200,
    background:'rgba(0,0,0,.4)',
    backdropFilter:'blur(6px)',
    display:'flex', alignItems:'center', justifyContent:'center',
  };

  const box = {
    background:'#fff',
    border:'1px solid var(--border)',
    borderRadius:16,
    padding:36,
    width:520, maxWidth:'94vw',
    boxShadow:'0 20px 60px rgba(0,0,0,.2)',
    animation:'popUp .3s ease',
  };

  const phase = st.phase || '';
  const p1done   = phase==='finetune'||phase==='done';
  const p1active = phase==='pretrain';
  const p2active = phase==='finetune';
  const p2done   = phase==='done';

  const phasePill = (active, done_, label) => ({
    flex:1, padding:'8px 12px', borderRadius:8,
    textAlign:'center', fontSize:12, fontWeight:600,
    background: done_ ? '#f0fdf4' : active ? 'var(--saf-lt)' : 'var(--warm)',
    border: done_ ? '1px solid #86efac' : active
      ? '1px solid rgba(232,114,12,.4)' : '1px solid var(--border)',
    color: done_ ? 'var(--ok)' : active ? 'var(--saffron)' : 'var(--text3)',
  });

  return (
    <>
      <style>{`@keyframes popUp{from{opacity:0;transform:scale(.9) translateY(16px)}to{opacity:1;transform:scale(1) translateY(0)}}`}</style>
      <div style={overlay} onClick={e => e.target===e.currentTarget && onClose()}>
        <div style={box}>
          <div style={{ fontFamily:'Playfair Display', fontWeight:700,
                        fontSize:20, color:'var(--text)', marginBottom:6 }}>
            Training JCNN Model
          </div>
          <p style={{ fontSize:13, color:'var(--text3)', marginBottom:22,
                      lineHeight:1.6 }}>
            Phase 1 — Pretrain JGCN + VIE (λ₄ = 0, Eq.12)<br/>
            Phase 2 — Finetune with JWC classification (λ₄ = 1.0)
          </p>

          {/* Phase pills */}
          <div style={{ display:'flex', gap:8, marginBottom:18 }}>
            <div style={phasePill(p1active, p1done, 'Phase 1')}>
              {p1done ? 'Phase 1 done' : 'Phase 1: Pretrain VIE'}
            </div>
            <div style={phasePill(p2active, p2done, 'Phase 2')}>
              {p2done ? 'Phase 2 done' : 'Phase 2: Finetune JWC'}
            </div>
          </div>

          {/* Message */}
          <div style={{ fontSize:13, color:'var(--saffron)', fontWeight:500,
                        minHeight:20, marginBottom:12 }}>
            {st.message || 'Waiting…'}
          </div>

          {/* Progress bar */}
          <div style={{ height:8, background:'var(--border)',
                        borderRadius:4, overflow:'hidden', marginBottom:6 }}>
            <div style={{
              height:'100%', borderRadius:4,
              background:'linear-gradient(90deg,var(--maroon),var(--saffron),var(--gold))',
              width:`${st.progress||0}%`,
              transition:'width .6s cubic-bezier(.22,1,.36,1)',
              boxShadow:'0 0 8px rgba(232,114,12,.4)',
            }}/>
          </div>
          <div style={{ fontSize:12, color:'var(--text3)',
                        textAlign:'right', marginBottom:20 }}>
            {st.progress||0}%
          </div>

          {/* Accuracy */}
          <div style={{
            textAlign:'center', padding:'16px 20px',
            background:'var(--warm)', border:'1px solid var(--border)',
            borderRadius:10, marginBottom:22,
          }}>
            <div style={{ fontFamily:'Playfair Display', fontWeight:700,
                          fontSize:36, color:'var(--saffron)' }}>
              {st.accuracy > 0 ? `${st.accuracy.toFixed(1)}%` : '—'}
            </div>
            <div style={{ fontSize:11, color:'var(--text3)', letterSpacing:2,
                          textTransform:'uppercase', marginTop:3 }}>
              Validation Accuracy
            </div>
          </div>

          {/* Buttons */}
          <div style={{ display:'flex', gap:10 }}>
            <button
              disabled={!done}
              onClick={onDone}
              style={{
                flex:1, padding:13,
                background: done ? 'var(--saffron)' : 'rgba(232,114,12,.3)',
                border:'none', borderRadius:10, color:'#fff',
                fontFamily:'Inter', fontWeight:700, fontSize:14,
                cursor: done ? 'pointer' : 'not-allowed',
              }}
            >
              {done ? 'Done — Start Recognising' : 'Training in progress…'}
            </button>
            <button onClick={onClose} style={{
              padding:'13px 20px',
              background:'var(--warm)', border:'1px solid var(--border)',
              borderRadius:10, fontFamily:'Inter', fontSize:14,
              color:'var(--text2)', cursor:'pointer',
            }}>
              Close
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
