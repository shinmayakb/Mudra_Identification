import React, { useRef, useState, useEffect, useCallback } from 'react';

// MediaPipe hand connection pairs (0–20 per hand)
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

const H0_JOINT = '#16a34a';   // left hand joint (green)
const H0_CONN  = '#4ade8080'; // left hand connection
const H1_JOINT = '#2563eb';   // right hand joint (blue)
const H1_CONN  = '#60a5fa80'; // right hand connection

function drawHandMesh(ctx, joints42, W, H) {
  if (!joints42 || joints42.length === 0) return;

  for (let hand = 0; hand < 2; hand++) {
    const off   = hand * 21;
    const jc    = hand === 0 ? H0_JOINT : H1_JOINT;
    const cc    = hand === 0 ? H0_CONN  : H1_CONN;

    // Check hand is present (not all zeros)
    const slice = Array.isArray(joints42[off])
      ? joints42.slice(off, off + 21)
      : null;
    const hasData = Array.isArray(joints42[0])
      ? joints42[off].some(v => v !== 0)
      : joints42.slice(off * 3, off * 3 + 3).some(v => v !== 0);
    if (!hasData) continue;

    const pt = (i) => {
      if (Array.isArray(joints42[0])) {
        const j = joints42[off + i];
        return { x: j[0] * W, y: j[1] * H };
      }
      return {
        x: joints42[(off + i) * 3]     * W,
        y: joints42[(off + i) * 3 + 1] * H,
      };
    };

    // Draw connections
    ctx.strokeStyle = cc;
    ctx.lineWidth   = 2;
    CONNECTIONS.forEach(([a, b]) => {
      const pa = pt(a), pb = pt(b);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    });

    // Draw joints as circles
    for (let j = 0; j < 21; j++) {
      const p = pt(j);
      // Fingertip joints larger
      const r = [4, 8, 12, 16, 20].includes(j) ? 5 : 3;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle   = jc;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth   = 1.5;
      ctx.fill();
      ctx.stroke();
    }
  }
}

export default function WebcamPanel({ API, onPrediction }) {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef  = useRef(null);

  const [running,  setRunning]  = useState(false);
  const [liveRes,  setLiveRes]  = useState(null);
  const [loading,  setLoading]  = useState(false);

  const captureMirrorFrame = useCallback(() => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return null;
    const c   = document.createElement('canvas');
    c.width   = v.videoWidth;
    c.height  = v.videoHeight;
    const ctx = c.getContext('2d');
    ctx.translate(c.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(v, 0, 0);
    return c.toDataURL('image/jpeg', 0.85);
  }, []);

  const drawOverlay = useCallback((joints) => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if (joints) drawHandMesh(ctx, joints, c.width, c.height);
  }, []);

  const startLiveLoop = useCallback(() => {
    timerRef.current = setInterval(async () => {
      if (!streamRef.current) return;
      const b64 = captureMirrorFrame();
      if (!b64) return;
      try {
        const r = await fetch(`${API}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: b64, source: 'webcam_live' }),
        });
        const d = await r.json();
        if (d.hand_detected && d.mudra) {
          setLiveRes({ name: d.mudra, conf: d.confidence_pct,
                       hands: d.hand_count, labels: d.hand_labels });
          drawOverlay(d.joints);
          onPrediction(d);
        } else {
          setLiveRes(null);
          drawOverlay(null);
        }
      } catch { /* ignore */ }
    }, 1200);
  }, [API, captureMirrorFrame, drawOverlay, onPrediction]);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
      });
      streamRef.current           = stream;
      videoRef.current.srcObject  = stream;
      setRunning(true);
      videoRef.current.onloadedmetadata = () => {
        const c   = canvasRef.current;
        if (c) {
          c.width  = videoRef.current.videoWidth;
          c.height = videoRef.current.videoHeight;
        }
        startLiveLoop();
      };
    } catch {
      alert('Camera access denied. Please allow camera and reload.');
    }
  }, [startLiveLoop]);

  const stopCamera = useCallback(() => {
    clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    setRunning(false);
    setLiveRes(null);
    drawOverlay(null);
  }, [drawOverlay]);

  const handleCapture = useCallback(async () => {
    clearInterval(timerRef.current);
    const b64 = captureMirrorFrame();
    if (!b64) return;
    setLoading(true);
    try {
      const r = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64, source: 'webcam_capture' }),
      });
      const d = await r.json();
      onPrediction(d);
      if (d.joints) drawOverlay(d.joints);
    } catch (e) {
      onPrediction({ error: String(e) });
    } finally {
      setLoading(false);
      startLiveLoop();
    }
  }, [API, captureMirrorFrame, drawOverlay, onPrediction, startLiveLoop]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const s = {
    wrap: {
      background: '#fff',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius)',
      overflow: 'hidden',
      boxShadow: 'var(--shadow)',
    },
    videoWrap: {
      position: 'relative',
      aspectRatio: '4/3',
      background: '#f0f0f0',
      overflow: 'hidden',
    },
    video: {
      position: 'absolute', inset: 0,
      width: '100%', height: '100%',
      objectFit: 'cover',
      transform: 'scaleX(-1)',
    },
    canvas: {
      position: 'absolute', inset: 0,
      width: '100%', height: '100%',
      transform: 'scaleX(-1)',
      pointerEvents: 'none',
    },
    overlay: {
      position: 'absolute', top: 12, left: '50%',
      transform: 'translateX(-50%)',
      background: 'rgba(255,255,255,.92)',
      backdropFilter: 'blur(8px)',
      border: '1px solid var(--border)',
      borderRadius: 40,
      padding: '7px 20px',
      textAlign: 'center',
      display: liveRes ? 'block' : 'none',
      whiteSpace: 'nowrap',
      boxShadow: 'var(--shadow)',
      zIndex: 10,
    },
    stopped: {
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      aspectRatio: '4/3',
      background: 'var(--warm)',
      gap: 16,
    },
    startBtn: {
      background: 'var(--saffron)', border: 'none', color: '#fff',
      padding: '12px 32px', borderRadius: 8,
      fontFamily: 'Inter', fontSize: 14, fontWeight: 600,
      cursor: 'pointer',
    },
    controls: {
      display: 'flex', gap: 10, padding: '12px 14px',
      borderTop: '1px solid var(--border)',
      background: 'var(--warm)',
    },
    btn: (primary) => ({
      flex: primary ? 2 : 1, padding: '11px 0',
      background: primary ? 'var(--saffron)' : '#fff',
      border: primary ? 'none' : '1px solid var(--border)',
      color: primary ? '#fff' : 'var(--text2)',
      borderRadius: 8, fontSize: 14, fontWeight: 600,
      cursor: 'pointer', fontFamily: 'Inter',
      opacity: loading ? 0.7 : 1,
    }),
  };

  if (!running) {
    return (
      <div style={s.wrap}>
        <div style={s.stopped}>
          <svg width="52" height="52" viewBox="0 0 24 24" fill="none"
               stroke="var(--text3)" strokeWidth="1.5" strokeLinecap="round">
            <path d="M15 10l4.553-2.069A1 1 0 0121 8.87v6.26a1 1 0 01-1.447.89L15 14"/>
            <rect x="2" y="7" width="13" height="10" rx="2"/>
          </svg>
          <p style={{ fontSize: 14, color: 'var(--text3)' }}>
            Camera is stopped
          </p>
          <button style={s.startBtn} onClick={startCamera}>
            Start Camera
          </button>
          <p style={{ fontSize: 12, color: 'var(--text3)', textAlign: 'center',
                      maxWidth: 260, lineHeight: 1.6 }}>
            Hand skeleton web will appear over your hands in real time.
            Works with single or both hands.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={s.wrap}>
      <div style={s.videoWrap}>
        <video ref={videoRef} style={s.video} autoPlay muted playsInline/>
        <canvas ref={canvasRef} style={s.canvas}/>
        {liveRes && (
          <div style={s.overlay}>
            <div style={{ fontFamily: 'Playfair Display', fontWeight: 700,
                          fontSize: 16, color: 'var(--text)' }}>
              {liveRes.name}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text3)', marginTop: 1 }}>
              {liveRes.conf}% — {liveRes.hands === 2 ? 'Both hands' : '1 hand'}
              {liveRes.labels?.length > 0 && ` (${liveRes.labels.join(', ')})`}
            </div>
          </div>
        )}
      </div>
      <div style={s.controls}>
        <button style={s.btn(true)} onClick={handleCapture} disabled={loading}>
          {loading ? 'Processing…' : 'Capture and Recognise'}
        </button>
        <button style={s.btn(false)} onClick={stopCamera}>
          Stop Camera
        </button>
      </div>
    </div>
  );
}
