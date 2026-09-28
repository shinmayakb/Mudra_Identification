import React, { useState } from 'react';
import WebcamPanel   from './WebcamPanel';
import { UploadPanel } from './UploadPanel';
import ResultPanel   from './ResultPanel';

export default function RecognizeSection({ API, modelReady, onResult, result, onTrainClick }) {
  const [tab, setTab] = useState('upload');

  return (
    <section id="recognize" style={{
      background: 'var(--warm)',
      borderBottom: '1px solid var(--border)',
      padding: '64px 48px',
    }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>

        {/* Section header */}
        <div style={{ textAlign: 'center', marginBottom: 40 }}>
          <div style={{ width:48, height:3, background:'var(--saffron)',
                        margin:'0 auto 16px', borderRadius:2 }}/>
          <h2 style={{
            fontFamily: 'Playfair Display', fontWeight:700,
            fontSize: 'clamp(1.5rem,3vw,2.2rem)',
            color: 'var(--text)', marginBottom:10,
          }}>
            Mudra Recognition
          </h2>
          <p style={{ fontSize:14, color:'var(--text2)', maxWidth:540,
                      margin:'0 auto', lineHeight:1.8 }}>
            Upload a hand gesture photo or use your live camera.
            The JCNN model detects single and dual hand mudras
            and draws the joint skeleton web on your hands.
          </p>
        </div>

        {/* Model not ready banner */}
        {!modelReady && (
          <div style={{
            background: '#fffbeb', border:'1px solid #fcd34d',
            borderRadius: 10, padding:'14px 20px',
            marginBottom:24, display:'flex', alignItems:'center',
            justifyContent:'space-between', gap:12,
          }}>
            <span style={{ fontSize:14, color:'#92400e' }}>
              Model not trained yet. Place your mudra images in
              <code style={{ background:'#fef3c7', padding:'2px 6px',
                              borderRadius:4, margin:'0 4px' }}>
                backend/dataset/
              </code>
              and train first.
            </span>
            <button onClick={onTrainClick} style={{
              background:'var(--saffron)', border:'none', color:'#fff',
              padding:'8px 18px', borderRadius:7, fontFamily:'Inter',
              fontSize:13, fontWeight:600, cursor:'pointer', flexShrink:0,
            }}>
              Train Now
            </button>
          </div>
        )}

        {/* Mode tabs */}
        <div style={{
          display:'flex', gap:4,
          background:'var(--border)', borderRadius:10,
          padding:4, width:'fit-content', margin:'0 auto 32px',
        }}>
          {[['upload','Upload Image'],['camera','Live Camera']].map(([k,label]) => (
            <button key={k} onClick={() => setTab(k)} style={{
              background: tab===k ? '#fff' : 'transparent',
              border:'none', color: tab===k ? 'var(--text)' : 'var(--text3)',
              padding:'9px 28px', borderRadius:8,
              fontFamily:'Inter', fontSize:14, fontWeight:600,
              cursor:'pointer', transition:'all .2s',
              boxShadow: tab===k ? 'var(--shadow)' : 'none',
            }}>
              {label}
            </button>
          ))}
        </div>

        {/* Grid */}
        <div style={{
          display:'grid',
          gridTemplateColumns: 'minmax(0,1.2fr) minmax(0,1fr)',
          gap:20,
        }}>
          <div>
            {tab === 'upload'
              ? <UploadPanel API={API} onPrediction={onResult}/>
              : <WebcamPanel API={API} onPrediction={onResult}/>
            }
          </div>
          <ResultPanel result={result}/>
        </div>

      </div>
    </section>
  );
}
