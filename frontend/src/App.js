import React, { useState, useEffect, useCallback, useRef } from 'react';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/* ─────────────────────────────────────────────────────────────────────────
   HAND MESH DRAWING  (21 joints per hand, both hands overlay on canvas)
───────────────────────────────────────────────────────────────────────── */
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

function drawHandMesh(ctx, joints, W, H) {
  if (!joints || !joints.length) return;
  for (let hand = 0; hand < 2; hand++) {
    const off = hand * 21;
    const isNested = Array.isArray(joints[0]);
    const hasData  = isNested
      ? joints[off] && joints[off].some(v => v !== 0)
      : joints.slice(off * 3, off * 3 + 3).some(v => v !== 0);
    if (!hasData) continue;
    const jc = hand === 0 ? '#16a34a' : '#2563eb';
    const cc = hand === 0 ? '#4ade8088' : '#60a5fa88';
    const pt = i => isNested
      ? { x: joints[off+i][0]*W, y: joints[off+i][1]*H }
      : { x: joints[(off+i)*3]*W, y: joints[(off+i)*3+1]*H };
    ctx.strokeStyle = cc; ctx.lineWidth = 2;
    CONNECTIONS.forEach(([a,b]) => {
      const pa=pt(a),pb=pt(b);
      ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
    });
    for (let j=0; j<21; j++) {
      const p=pt(j), r=[4,8,12,16,20].includes(j)?5:3;
      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2);
      ctx.fillStyle=jc; ctx.strokeStyle='#fff'; ctx.lineWidth=1.5;
      ctx.fill(); ctx.stroke();
    }
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   HEADER
───────────────────────────────────────────────────────────────────────── */
function Header({ modelReady, numClasses, onTrain }) {
  return (
    <header style={{ background:'#fff', borderBottom:'1px solid var(--border)',
      padding:'0 40px', height:64, display:'flex', alignItems:'center',
      justifyContent:'space-between', position:'sticky', top:0, zIndex:100,
      boxShadow:'0 1px 4px rgba(0,0,0,.06)' }}>
      <div style={{ display:'flex', alignItems:'center', gap:12 }}>
        <div style={{ width:36, height:36, background:'linear-gradient(135deg,var(--saffron),var(--gold))',
          borderRadius:8, display:'flex', alignItems:'center', justifyContent:'center' }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round">
            <path d="M18 10V6a2 2 0 00-4 0v4"/><path d="M14 6V4a2 2 0 00-4 0v6"/>
            <path d="M10 5V4a2 2 0 00-4 0v10"/><path d="M6 14v-3a2 2 0 014 0"/>
            <rect x="2" y="14" width="20" height="8" rx="2"/>
          </svg>
        </div>
        <div>
          <div style={{ fontFamily:'Playfair Display', fontWeight:700, fontSize:16, color:'var(--text)' }}>
            Mudra Recognition
          </div>
          <div style={{ fontSize:10, color:'var(--text3)', letterSpacing:2, textTransform:'uppercase' }}>
            Bharatanatyam
          </div>
        </div>
      </div>
      <div style={{ display:'flex', alignItems:'center', gap:14 }}>
        <div style={{ display:'flex', alignItems:'center', gap:6, fontSize:13, color:'var(--text2)' }}>
          <div style={{ width:8, height:8, borderRadius:'50%',
            background: modelReady ? 'var(--ok)' : 'var(--err)',
            boxShadow: modelReady ? '0 0 6px #15803d80' : '0 0 6px #b91c1c80' }}/>
          {modelReady ? `Model ready (${numClasses} classes)` : 'Model not trained'}
        </div>
        <button onClick={onTrain} style={{ background:'var(--saffron)', border:'none', color:'#fff',
          padding:'8px 20px', borderRadius:7, fontFamily:'Inter', fontSize:13, fontWeight:600,
          cursor:'pointer' }}>
          Train Model
        </button>
      </div>
    </header>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   HERO SECTION
───────────────────────────────────────────────────────────────────────── */
function HeroSection() {
  return (
    <section style={{ background:'linear-gradient(135deg,#FFF8F0,#FEF3E8,#F9EDD8)',
      borderBottom:'1px solid var(--border)', padding:'80px 40px',
      textAlign:'center', position:'relative', overflow:'hidden' }}>
      {[200,340,480].map((size,i) => (
        <div key={i} style={{ position:'absolute', top:'50%', left:'50%',
          width:size, height:size, borderRadius:'50%',
          border:`1px solid rgba(232,114,12,${0.06-i*0.015})`,
          transform:'translate(-50%,-50%)', pointerEvents:'none' }}/>
      ))}
      <div style={{ position:'relative', maxWidth:720, margin:'0 auto' }}>
        <div style={{ display:'inline-block', background:'rgba(232,114,12,.1)',
          border:'1px solid rgba(232,114,12,.3)', color:'var(--saffron)',
          padding:'5px 18px', borderRadius:40, fontSize:11, fontWeight:600,
          letterSpacing:3, textTransform:'uppercase', marginBottom:22 }}>
          IEEE TMM 2025 — VIE + JWC Architecture
        </div>
        <h1 style={{ fontFamily:'Playfair Display', fontWeight:700,
          fontSize:'clamp(2rem,5vw,3.4rem)', color:'var(--text)', lineHeight:1.2, marginBottom:16 }}>
          Bharatanatyam Mudra<br/>
          <span style={{ color:'var(--saffron)' }}>Recognition System</span>
        </h1>
        <p style={{ fontSize:16, color:'var(--text2)', lineHeight:1.8,
          maxWidth:600, margin:'0 auto 32px' }}>
          Recognise classical Indian hand gestures using Graph Convolutional Networks
          with Viewpoint Influence Elimination and Joint-Weighted Classification —
          works from any angle with single or dual hands.
        </p>
        <div style={{ display:'flex', gap:12, justifyContent:'center', flexWrap:'wrap' }}>
          {[['JGCN + KGCN','Graph Convolutional Networks'],['VIE','Viewpoint Invariance'],
            ['JWC','Joint-Weighted Classification'],['Dual Hand','Single & Both Hands']].map(([l,d])=>(
            <div key={l} style={{ background:'#fff', border:'1px solid var(--border)',
              borderRadius:10, padding:'10px 16px', textAlign:'center',
              boxShadow:'0 1px 4px rgba(0,0,0,.06)' }}>
              <div style={{ fontSize:13, fontWeight:600, color:'var(--saffron)' }}>{l}</div>
              <div style={{ fontSize:11, color:'var(--text3)', marginTop:2 }}>{d}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   ABOUT SECTION
───────────────────────────────────────────────────────────────────────── */
const MUDRAS_REF = [
  { name:'Pataka',      meaning:'Flag',               description:'Four fingers straight and together, thumb bent. Represents a flag, forest, night, moonlight.' },
  { name:'Tripataka',   meaning:'Three Parts of Flag', description:'Like Pataka but ring finger bent. Used for crown, tree, arrow, lightning.' },
  { name:'Alapadma',    meaning:'Full-Bloomed Lotus',  description:'All fingers spread wide. Represents a lotus in full bloom, beauty, a globe.' },
  { name:'Mushti',      meaning:'Fist',                description:'All fingers closed in a fist. Represents firmness, stubbornness, fighting.' },
  { name:'Shikhara',    meaning:'Spire',               description:'Thumb erect, fingers closed. Represents Shiva, a spire, a bow, love.' },
  { name:'Katakamukha', meaning:'Opening of a Link',   description:'Index and middle finger together with thumb. Used for pearl strings, pulling a bowstring.' },
  { name:'Mayura',      meaning:'Peacock',             description:'Thumb and index form a circle, others extended. Represents a peacock, eyebrow, neck of a bird.' },
  { name:'Chandrakala', meaning:'Moon',                description:'Modified Ardhachandra. Represents the moon, nectar, Shiva\'s forehead.' },
];

function AboutSection() {
  const [open, setOpen] = useState(false);
  const cards = [
    { title:'Asamyuta Mudras', sub:'Single-hand gestures', accent:'var(--saffron)',
      text:'Performed with one hand, these 28 mudras each carry distinct symbolic meaning. From Pataka (flag) to Mushti (fist), each finger position is precisely prescribed in the Natya Shastra — the ancient Sanskrit treatise on performing arts.' },
    { title:'Samyuta Mudras', sub:'Both-hand gestures', accent:'var(--maroon)',
      text:'Using both hands together, these 24 mudras create more complex meanings — depicting deities, natural phenomena, and abstract concepts. Our system detects both hands simultaneously to recognise these combined gestures.' },
    { title:'Natya Shastra', sub:'The ancient treatise', accent:'var(--gold)',
      text:'Attributed to Bharata Muni (200 BCE – 200 CE), the Natya Shastra codifies 108 karanas, 64 principles of dance, and precisely describes every hand gesture with its meaning and application in performance.' },
    { title:'Symbolic Language', sub:'Meaning through gesture', accent:'var(--teal)',
      text:'A single mudra can convey dozens of meanings depending on context. Pataka alone can mean flag, wind, forest, waves, honour, night, river, or entering — context makes the difference.' },
  ];
  return (
    <section style={{ background:'#fff', borderBottom:'1px solid var(--border)', padding:'64px 40px' }}>
      <div style={{ maxWidth:1100, margin:'0 auto' }}>
        <div style={{ textAlign:'center', marginBottom:48 }}>
          <div style={{ width:48, height:3, background:'var(--saffron)', margin:'0 auto 16px', borderRadius:2 }}/>
          <h2 style={{ fontFamily:'Playfair Display', fontWeight:700,
            fontSize:'clamp(1.6rem,3vw,2.4rem)', color:'var(--text)', marginBottom:12 }}>
            What is Mudra in Bharatanatyam?
          </h2>
          <p style={{ fontSize:15, color:'var(--text2)', maxWidth:680, margin:'0 auto', lineHeight:1.8 }}>
            In Bharatanatyam, <em>mudras</em> (literally "seal" or "sign") are codified hand gestures
            that communicate emotions, narratives, and symbolic meanings within classical Indian dance.
            They form the vocabulary of <em>abhinaya</em> — the art of expression.
          </p>
        </div>
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(280px,1fr))', gap:22, marginBottom:48 }}>
          {cards.map(c=>(
            <div key={c.title} style={{ background:'var(--warm)', border:'1px solid var(--border)',
              borderRadius:'var(--radius)', padding:'22px 20px', borderTop:`3px solid ${c.accent}` }}>
              <div style={{ fontWeight:600, fontSize:16, color:'var(--text)', marginBottom:4 }}>{c.title}</div>
              <div style={{ fontSize:12, color:c.accent, fontWeight:600, letterSpacing:1,
                textTransform:'uppercase', marginBottom:12 }}>{c.sub}</div>
              <p style={{ fontSize:14, color:'var(--text2)', lineHeight:1.7 }}>{c.text}</p>
            </div>
          ))}
        </div>
        <div style={{ background:'var(--warm)', border:'1px solid var(--border)',
          borderRadius:'var(--radius)', overflow:'hidden' }}>
          <button onClick={()=>setOpen(o=>!o)} style={{ width:'100%', background:'none', border:'none',
            padding:'16px 20px', display:'flex', alignItems:'center', justifyContent:'space-between',
            cursor:'pointer', fontFamily:'Inter' }}>
            <span style={{ fontWeight:600, fontSize:15, color:'var(--text)' }}>
              Common Bharatanatyam Mudras — Quick Reference
            </span>
            <span style={{ fontSize:18, color:'var(--saffron)',
              transform:open?'rotate(180deg)':'none', transition:'transform .25s', display:'inline-block' }}>
              ▼
            </span>
          </button>
          {open && (
            <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(240px,1fr))',
              gap:1, borderTop:'1px solid var(--border)' }}>
              {MUDRAS_REF.map(m=>(
                <div key={m.name} style={{ padding:'14px 18px', background:'#fff',
                  borderBottom:'1px solid var(--border)' }}>
                  <div style={{ fontWeight:600, color:'var(--text)', fontSize:14, marginBottom:2 }}>
                    {m.name}
                    <span style={{ fontSize:12, color:'var(--saffron)', marginLeft:8, fontWeight:400 }}>
                      {m.meaning}
                    </span>
                  </div>
                  <div style={{ fontSize:12, color:'var(--text3)', lineHeight:1.6 }}>{m.description}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   UPLOAD PANEL
───────────────────────────────────────────────────────────────────────── */
function UploadPanel({ onPrediction }) {
  const inputRef  = useRef(null);
  const [b64,     setB64]     = useState(null);
  const [annB64,  setAnnB64]  = useState(null);
  const [drag,    setDrag]    = useState(false);
  const [loading, setLoading] = useState(false);

  const loadFile = f => {
    if (!f || !f.type.startsWith('image/')) return;
    const r = new FileReader();
    r.onload = e => { setB64(e.target.result); setAnnB64(null); };
    r.readAsDataURL(f);
  };
  const predict = async () => {
    if (!b64) return;
    setLoading(true);
    try {
      const r = await fetch(`${API}/predict`, { method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ image:b64, source:'upload' }) });
      const d = await r.json();
      onPrediction(d);
      if (d.drawn_image) setAnnB64(d.drawn_image);
    } catch(e) { onPrediction({ error:String(e) }); }
    finally { setLoading(false); }
  };
  const clear = () => {
    setB64(null); setAnnB64(null);
    if (inputRef.current) inputRef.current.value='';
    onPrediction(null);
  };
  return (
    <div style={{ background:'#fff', border:'1px solid var(--border)', borderRadius:'var(--radius)',
      overflow:'hidden', boxShadow:'var(--shadow)' }}>
      <div style={{ position:'relative', aspectRatio:'4/3', margin:14, borderRadius:10,
        display:'flex', alignItems:'center', justifyContent:'center', overflow:'hidden',
        background:drag?'var(--saf-lt)':b64?'#f9f9f9':'var(--warm)',
        border:drag?'2px dashed var(--saffron)':'2px dashed var(--border)',
        cursor:b64?'default':'pointer', transition:'all .2s' }}
        onDragOver={e=>{e.preventDefault();setDrag(true)}}
        onDragLeave={()=>setDrag(false)}
        onDrop={e=>{e.preventDefault();setDrag(false);loadFile(e.dataTransfer.files[0])}}
        onClick={()=>!b64&&inputRef.current?.click()}>
        {!b64 ? (
          <div style={{ textAlign:'center', padding:32 }}>
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none"
                 stroke="var(--text3)" strokeWidth="1.5" style={{ marginBottom:12 }}>
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <div style={{ fontWeight:600, fontSize:15, color:'var(--text)', marginBottom:6 }}>
              Drop image here
            </div>
            <div style={{ fontSize:13, color:'var(--text3)', lineHeight:1.6 }}>
              or click to browse<br/>JPG, PNG, WebP, BMP
            </div>
            <div style={{ marginTop:14, fontSize:12, color:'var(--saffron)',
              background:'var(--saf-lt)', padding:'6px 14px', borderRadius:20, display:'inline-block' }}>
              Plain background recommended
            </div>
          </div>
        ) : (
          <>
            <img src={b64} alt="preview" style={{ position:'absolute',inset:0,width:'100%',height:'100%',objectFit:'contain' }}/>
            {annB64 && <img src={annB64} alt="annotated" style={{ position:'absolute',inset:0,width:'100%',height:'100%',objectFit:'contain',zIndex:2 }}/>}
          </>
        )}
      </div>
      <input ref={inputRef} type="file" accept="image/*" style={{ display:'none' }}
             onChange={e=>loadFile(e.target.files[0])}/>
      <div style={{ display:'flex', gap:10, padding:'0 14px 14px' }}>
        <button disabled={!b64||loading} onClick={predict} style={{
          flex:2, padding:'11px 0',
          background:(!b64||loading)?'#ccc':'var(--saffron)',
          border:'none', color:'#fff', borderRadius:8, fontSize:14,
          fontWeight:600, cursor:(!b64||loading)?'not-allowed':'pointer', fontFamily:'Inter' }}>
          {loading?'Recognising…':'Recognise Mudra'}
        </button>
        <button onClick={clear} style={{ flex:1, padding:'11px 0', background:'#fff',
          border:'1px solid var(--border)', color:'var(--text2)', borderRadius:8,
          fontSize:14, fontWeight:600, cursor:'pointer', fontFamily:'Inter' }}>
          Clear
        </button>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   WEBCAM PANEL (FIXED VERSION)
───────────────────────────────────────────────────────────────────────── */
function WebcamPanel({ onPrediction }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [liveRes, setLiveRes] = useState(null);
  const [loading, setLoading] = useState(false);

  /* Capture frame */
  const capFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return null;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    /* mirror correct */
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    ctx.drawImage(video, 0, 0);

    return canvas.toDataURL("image/jpeg", 0.85);
  }, []);

  /* Draw joints */
  const drawOv = useCallback((joints) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (joints) {
      drawHandMesh(ctx, joints, canvas.width, canvas.height);
    }
  }, []);

  /* Live prediction loop */
  const startLoop = useCallback(() => {
    clearInterval(timerRef.current);

    timerRef.current = setInterval(async () => {
      if (!streamRef.current) return;

      const img = capFrame();
      if (!img) return;

      try {
        const res = await fetch(`${API}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: img,
            source: "webcam_live"
          })
        });

        const data = await res.json();

        if (data.hand_detected && data.mudra) {
          setLiveRes({
            name: data.mudra,
            conf: data.confidence_pct,
            hands: data.hand_count
          });

          drawOv(data.joints);
          onPrediction(data);
        } else {
          setLiveRes(null);
          drawOv(null);
        }

      } catch (err) {}
    }, 1200);
  }, [capFrame, drawOv, onPrediction]);

  /* Start camera */
  const startCam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });

      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;

      video.onloadedmetadata = async () => {
        await video.play();

        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        setRunning(true);
        startLoop();
      };

    } catch (err) {
      alert("Camera not allowed or not found");
    }
  }, [startLoop]);

  /* Stop camera */
  const stopCam = useCallback(() => {
    clearInterval(timerRef.current);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setRunning(false);
    setLiveRes(null);
    drawOv(null);
  }, [drawOv]);

  /* Capture */
  const capture = useCallback(async () => {
    const img = capFrame();
    if (!img) return;

    setLoading(true);

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: img,
          source: "webcam_capture"
        })
      });

      const data = await res.json();

      onPrediction(data);

      if (data.joints) {
        drawOv(data.joints);
      }

    } catch (err) {}

    setLoading(false);
  }, [capFrame, drawOv, onPrediction]);

  useEffect(() => {
    return () => stopCam();
  }, [stopCam]);

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        overflow: "hidden",
        boxShadow: "var(--shadow)"
      }}
    >
      {/* Camera Area */}
      <div
        style={{
          position: "relative",
          background: "#000",
          minHeight: 420
        }}
      >
        {/* Real Video */}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{
            width: "100%",
            display: running ? "block" : "none",
            transform: "scaleX(-1)"
          }}
        />

        {/* Overlay */}
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            transform: "scaleX(-1)",
            display: running ? "block" : "none"
          }}
        />

        {/* Placeholder */}
        {!running && (
          <div
            style={{
              minHeight: 420,
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
              color: "#777",
              gap: 15
            }}
          >
            <p>Camera is stopped</p>

            <button
              onClick={startCam}
              style={{
                background: "var(--saffron)",
                border: "none",
                color: "#fff",
                padding: "12px 30px",
                borderRadius: 8,
                cursor: "pointer"
              }}
            >
              Start Camera
            </button>
          </div>
        )}

        {/* Live badge */}
        {running && liveRes && (
          <div
            style={{
              position: "absolute",
              top: 12,
              left: "50%",
              transform: "translateX(-50%)",
              background: "#fff",
              padding: "8px 18px",
              borderRadius: 30,
              fontSize: 14,
              fontWeight: "bold"
            }}
          >
            {liveRes.name} ({liveRes.conf}%)
          </div>
        )}
      </div>

      {/* Controls */}
      {running && (
        <div
          style={{
            display: "flex",
            gap: 10,
            padding: 14
          }}
        >
          <button
            onClick={capture}
            disabled={loading}
            style={{
              flex: 2,
              padding: 12,
              background: "var(--saffron)",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              cursor: "pointer"
            }}
          >
            {loading ? "Processing..." : "Capture and Recognise"}
          </button>

          <button
            onClick={stopCam}
            style={{
              flex: 1,
              padding: 12,
              background: "#fff",
              border: "1px solid #ccc",
              borderRadius: 8,
              cursor: "pointer"
            }}
          >
            Stop Camera
          </button>
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   RESULT PANEL
───────────────────────────────────────────────────────────────────────── */
function Card({ title, children, accent }) {
  return (
    <div style={{ background:'#fff', border:'1px solid var(--border)', borderRadius:'var(--radius)',
      padding:'18px 20px', borderTop:`3px solid ${accent||'var(--saffron)'}`,
      boxShadow:'var(--shadow)' }}>
      <div style={{ fontSize:11, fontWeight:600, letterSpacing:2, textTransform:'uppercase',
        color:'var(--text3)', marginBottom:14 }}>{title}</div>
      {children}
    </div>
  );
}

function ResultPanel({ result }) {
  if (!result) return (
    <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
      <Card title="Prediction Result">
        <div style={{ textAlign:'center', padding:'20px 0', color:'var(--text3)' }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="1.5" style={{ marginBottom:10 }}>
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <p style={{ fontSize:14, lineHeight:1.7 }}>
            Upload a hand gesture image<br/>or start the camera to begin
          </p>
        </div>
      </Card>
      <Card title="Top Candidates">
        <p style={{ fontSize:13, color:'var(--text3)', padding:'8px 0' }}>
          Candidates appear after recognition
        </p>
      </Card>
    </div>
  );

  if (result.error&&!result.hand_detected&&!result.mudra) return (
    <Card title="Result" accent="var(--err)">
      <div style={{ background:'#fef2f2', border:'1px solid #fecaca', borderRadius:8,
        padding:'12px 14px', fontSize:14, color:'var(--err)', lineHeight:1.6 }}>
        {result.error}
      </div>
    </Card>
  );

  const conf=result.confidence_pct??(result.confidence*100);
  const top5=result.top5||[];
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
      <Card title="Prediction Result" accent="var(--saffron)">
        <div style={{ textAlign:'center', padding:'8px 0 14px' }}>
          <div style={{ fontFamily:'Playfair Display', fontWeight:700,
            fontSize:32, color:'var(--text)', marginBottom:6, letterSpacing:.5 }}>
            {result.mudra||'—'}
          </div>
          <div style={{ display:'inline-block', background:'var(--saf-lt)',
            border:'1px solid rgba(232,114,12,.25)', color:'var(--saffron)',
            fontSize:11, fontWeight:600, letterSpacing:2, padding:'3px 14px',
            borderRadius:20, textTransform:'uppercase' }}>
            {result.hand_count===2?'Both hands':'1 hand detected'}
          </div>
        </div>
        <div style={{ marginTop:4 }}>
          <div style={{ display:'flex', justifyContent:'space-between', marginBottom:7, fontSize:13 }}>
            <span style={{ color:'var(--text3)' }}>Confidence</span>
            <span style={{ fontWeight:700, color:'var(--text)' }}>{conf.toFixed(1)}%</span>
          </div>
          <div style={{ height:7, background:'var(--border)', borderRadius:4, overflow:'hidden' }}>
            <div style={{ height:'100%', borderRadius:4, width:`${conf}%`, transition:'width .8s ease',
              background:conf>80?'linear-gradient(90deg,var(--saffron),var(--gold))':
                conf>50?'linear-gradient(90deg,var(--gold),#d4a017)':'#ef4444' }}/>
          </div>
        </div>
      </Card>
      <Card title="Top Candidates">
        {top5.length===0
          ? <p style={{ fontSize:13, color:'var(--text3)' }}>No data</p>
          : top5.map((p,i)=>(
            <div key={i} style={{ display:'flex', alignItems:'center', gap:10,
              padding:'8px 4px', borderBottom:i<top5.length-1?'1px solid var(--border)':'none' }}>
              <span style={{ fontSize:11, color:'var(--text3)', width:22, textAlign:'right', flexShrink:0 }}>
                #{i+1}
              </span>
              <div style={{ flex:1, minWidth:0 }}>
                <div style={{ fontSize:14, fontWeight:600, color:'var(--text)', letterSpacing:.3 }}>
                  {p.name}
                </div>
              </div>
              <div style={{ width:72, flexShrink:0 }}>
                <div style={{ height:4, background:'var(--border)', borderRadius:2 }}>
                  <div style={{ height:'100%', borderRadius:2, width:`${p.confidence_pct}%`,
                    background:i===0?'var(--saffron)':'var(--border2)', transition:'width .5s' }}/>
                </div>
              </div>
              <span style={{ fontSize:12, color:'var(--text3)', width:40, textAlign:'right', flexShrink:0 }}>
                {p.confidence_pct.toFixed(1)}%
              </span>
            </div>
          ))}
      </Card>
      <Card title="Detection Info" accent="var(--teal)">
        {[['Hands detected',result.hand_count??'—'],
          ['Hand labels',result.hand_labels?.join(', ')||'—'],
          ['Joints extracted',result.joints?`${result.joints.length} joints`:'—']
         ].map(([k,v])=>(
          <div key={k} style={{ display:'flex', justifyContent:'space-between',
            fontSize:13, padding:'6px 0', borderBottom:'1px solid var(--border)' }}>
            <span style={{ color:'var(--text3)' }}>{k}</span>
            <span style={{ fontWeight:600, color:'var(--text)' }}>{v}</span>
          </div>
        ))}
      </Card>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   TRAIN MODAL
───────────────────────────────────────────────────────────────────────── */
function TrainModal({ onClose, onDone }) {
  const [st,setSt]=useState({ progress:0,message:'Starting…',accuracy:0,phase:'',running:true });
  const [done,setDone]=useState(false);
  const poll=useRef(null);

  useEffect(()=>{
    fetch(`${API}/train`,{method:'POST'}).catch(()=>{});
    const run=async()=>{
      try{
        const r=await fetch(`${API}/train/status`);
        const d=await r.json(); setSt(d);
        if(!d.running&&d.progress>=100){ clearInterval(poll.current); setDone(true); }
      }catch{}
    };
    run(); poll.current=setInterval(run,1500);
    return()=>clearInterval(poll.current);
  },[]);

  const ph=st.phase||'';
  const pill=(act,dn,lbl)=>({
    flex:1,padding:'8px 12px',borderRadius:8,textAlign:'center',fontSize:12,fontWeight:600,
    background:dn?'#f0fdf4':act?'var(--saf-lt)':'var(--warm)',
    border:dn?'1px solid #86efac':act?'1px solid rgba(232,114,12,.4)':'1px solid var(--border)',
    color:dn?'var(--ok)':act?'var(--saffron)':'var(--text3)',
  });

  return (
    <>
      <style>{`@keyframes popUp{from{opacity:0;transform:scale(.9) translateY(16px)}to{opacity:1;transform:scale(1) translateY(0)}}`}</style>
      <div style={{ position:'fixed',inset:0,zIndex:200,background:'rgba(0,0,0,.4)',
        backdropFilter:'blur(6px)',display:'flex',alignItems:'center',justifyContent:'center' }}
        onClick={e=>e.target===e.currentTarget&&onClose()}>
        <div style={{ background:'#fff',border:'1px solid var(--border)',borderRadius:16,
          padding:36,width:520,maxWidth:'94vw',boxShadow:'0 20px 60px rgba(0,0,0,.2)',
          animation:'popUp .3s ease' }}>
          <div style={{ fontFamily:'Playfair Display',fontWeight:700,fontSize:20,
            color:'var(--text)',marginBottom:6 }}>Training JCNN Model</div>
          <p style={{ fontSize:13,color:'var(--text3)',marginBottom:22,lineHeight:1.6 }}>
            Phase 1 — Pretrain JGCN + VIE (λ₄ = 0, Eq.12)<br/>
            Phase 2 — Finetune with JWC classification (λ₄ = 1.0)
          </p>
          <div style={{ display:'flex',gap:8,marginBottom:18 }}>
            <div style={pill(ph==='pretrain',ph==='finetune'||ph==='done')}>
              {ph==='finetune'||ph==='done'?'Phase 1 done':'Phase 1: Pretrain VIE'}
            </div>
            <div style={pill(ph==='finetune',ph==='done')}>
              {ph==='done'?'Phase 2 done':'Phase 2: Finetune JWC'}
            </div>
          </div>
          <div style={{ fontSize:13,color:'var(--saffron)',fontWeight:500,minHeight:20,marginBottom:12 }}>
            {st.message||'Waiting…'}
          </div>
          <div style={{ height:8,background:'var(--border)',borderRadius:4,overflow:'hidden',marginBottom:6 }}>
            <div style={{ height:'100%',borderRadius:4,width:`${st.progress||0}%`,
              background:'linear-gradient(90deg,var(--maroon),var(--saffron),var(--gold))',
              transition:'width .6s cubic-bezier(.22,1,.36,1)' }}/>
          </div>
          <div style={{ fontSize:12,color:'var(--text3)',textAlign:'right',marginBottom:20 }}>
            {st.progress||0}%
          </div>
          <div style={{ textAlign:'center',padding:'16px 20px',background:'var(--warm)',
            border:'1px solid var(--border)',borderRadius:10,marginBottom:22 }}>
            <div style={{ fontFamily:'Playfair Display',fontWeight:700,fontSize:36,color:'var(--saffron)' }}>
              {st.accuracy>0?`${st.accuracy.toFixed(1)}%`:'—'}
            </div>
            <div style={{ fontSize:11,color:'var(--text3)',letterSpacing:2,textTransform:'uppercase',marginTop:3 }}>
              Validation Accuracy
            </div>
          </div>
          <div style={{ display:'flex',gap:10 }}>
            <button disabled={!done} onClick={onDone} style={{ flex:1,padding:13,
              background:done?'var(--saffron)':'rgba(232,114,12,.3)',border:'none',
              borderRadius:10,color:'#fff',fontFamily:'Inter',fontWeight:700,fontSize:14,
              cursor:done?'pointer':'not-allowed' }}>
              {done?'Done — Start Recognising':'Training in progress…'}
            </button>
            <button onClick={onClose} style={{ padding:'13px 20px',background:'var(--warm)',
              border:'1px solid var(--border)',borderRadius:10,fontFamily:'Inter',
              fontSize:14,color:'var(--text2)',cursor:'pointer' }}>
              Close
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   RECOGNIZE SECTION
───────────────────────────────────────────────────────────────────────── */
function RecognizeSection({ modelReady, onResult, result, onTrainClick }) {
  const [tab, setTab] = useState('upload');
  return (
    <section style={{ background:'var(--warm)', padding:'64px 40px' }}>
      <div style={{ maxWidth:1100, margin:'0 auto' }}>
        <div style={{ textAlign:'center', marginBottom:40 }}>
          <div style={{ width:48,height:3,background:'var(--saffron)',margin:'0 auto 16px',borderRadius:2 }}/>
          <h2 style={{ fontFamily:'Playfair Display',fontWeight:700,
            fontSize:'clamp(1.5rem,3vw,2.2rem)',color:'var(--text)',marginBottom:10 }}>
            Mudra Recognition
          </h2>
          <p style={{ fontSize:14,color:'var(--text2)',maxWidth:540,margin:'0 auto',lineHeight:1.8 }}>
            Upload a hand gesture photo or use your live camera.
            The JCNN model detects single and dual hand mudras
            and draws the joint skeleton web on your hands.
          </p>
        </div>
        {!modelReady&&(
          <div style={{ background:'#fffbeb',border:'1px solid #fcd34d',borderRadius:10,
            padding:'14px 20px',marginBottom:24,display:'flex',alignItems:'center',
            justifyContent:'space-between',gap:12 }}>
            <span style={{ fontSize:14,color:'#92400e' }}>
              Model not trained. Place mudra images in
              <code style={{ background:'#fef3c7',padding:'2px 6px',borderRadius:4,margin:'0 4px' }}>
                backend/dataset/
              </code>
              and train first.
            </span>
            <button onClick={onTrainClick} style={{ background:'var(--saffron)',border:'none',
              color:'#fff',padding:'8px 18px',borderRadius:7,fontFamily:'Inter',
              fontSize:13,fontWeight:600,cursor:'pointer',flexShrink:0 }}>
              Train Now
            </button>
          </div>
        )}
        <div style={{ display:'flex',gap:4,background:'var(--border)',borderRadius:10,
          padding:4,width:'fit-content',margin:'0 auto 32px' }}>
          {[['upload','Upload Image'],['camera','Live Camera']].map(([k,lbl])=>(
            <button key={k} onClick={()=>setTab(k)} style={{
              background:tab===k?'#fff':'transparent',border:'none',
              color:tab===k?'var(--text)':'var(--text3)',padding:'9px 28px',
              borderRadius:8,fontFamily:'Inter',fontSize:14,fontWeight:600,
              cursor:'pointer',transition:'all .2s',
              boxShadow:tab===k?'var(--shadow)':'none' }}>
              {lbl}
            </button>
          ))}
        </div>
        <div style={{ display:'grid',gridTemplateColumns:'minmax(0,1.2fr) minmax(0,1fr)',gap:20 }}>
          <div>
            {tab==='upload'
              ? <UploadPanel onPrediction={onResult}/>
              : <WebcamPanel onPrediction={onResult}/>}
          </div>
          <ResultPanel result={result}/>
        </div>
      </div>
    </section>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   ROOT APP
───────────────────────────────────────────────────────────────────────── */
export default function App() {
  const [modelReady, setModelReady] = useState(false);
  const [numClasses, setNumClasses] = useState(0);
  const [showTrain,  setShowTrain]  = useState(false);
  const [result,     setResult]     = useState(null);

  useEffect(() => {
    const check = async () => {
      try {
        const r=await fetch(`${API}/health`);
        const d=await r.json();
        setModelReady(d.model_ready);
        setNumClasses(d.num_classes||0);
      } catch {}
    };
    check();
    const id=setInterval(check,6000);
    return()=>clearInterval(id);
  }, []);

  return (
    <div style={{ minHeight:'100vh', background:'var(--cream)' }}>
      <Header modelReady={modelReady} numClasses={numClasses}
              onTrain={()=>setShowTrain(true)}/>
      <HeroSection/>
      <AboutSection/>
      <RecognizeSection modelReady={modelReady} onResult={setResult}
                        result={result} onTrainClick={()=>setShowTrain(true)}/>
      {showTrain&&(
        <TrainModal
          onClose={()=>setShowTrain(false)}
          onDone={()=>{ setShowTrain(false);
            fetch(`${API}/health`).then(r=>r.json()).then(d=>{
              setModelReady(d.model_ready); setNumClasses(d.num_classes||0);
            }).catch(()=>{}); }}/>
      )}
    </div>
  );
}