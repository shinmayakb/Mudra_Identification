import React from 'react';

const Card = ({ title, children, accent }) => (
  <div style={{
    background: '#fff', border: '1px solid var(--border)',
    borderRadius: 'var(--radius)', padding: '18px 20px',
    borderTop: `3px solid ${accent || 'var(--saffron)'}`,
    boxShadow: 'var(--shadow)',
  }}>
    <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: 2,
                  textTransform: 'uppercase', color: 'var(--text3)',
                  marginBottom: 14 }}>
      {title}
    </div>
    {children}
  </div>
);

export default function ResultPanel({ result }) {
  if (!result) {
    return (
      <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
        <Card title="Prediction Result">
          <div style={{ textAlign:'center', padding:'20px 0',
                        color:'var(--text3)' }}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" strokeWidth="1.5" style={{ marginBottom:10 }}>
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
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
  }

  if (result.error && !result.hand_detected && !result.mudra) {
    return (
      <Card title="Result" accent="var(--err)">
        <div style={{
          background: '#fef2f2', border: '1px solid #fecaca',
          borderRadius: 8, padding: '12px 14px',
          fontSize: 14, color: 'var(--err)', lineHeight: 1.6,
        }}>
          {result.error}
        </div>
      </Card>
    );
  }

  const conf = result.confidence_pct ?? (result.confidence * 100);
  const top5 = result.top5 || [];

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:14 }}>

      {/* Main prediction */}
      <Card title="Prediction Result" accent="var(--saffron)">
        <div style={{ textAlign:'center', padding:'8px 0 14px' }}>
          <div style={{
            fontFamily: 'Playfair Display', fontWeight: 700,
            fontSize: 32, color: 'var(--text)',
            marginBottom: 6, letterSpacing: .5,
          }}>
            {result.mudra || '—'}
          </div>
          <div style={{
            display:'inline-block',
            background: 'var(--saf-lt)',
            border: '1px solid rgba(232,114,12,.25)',
            color: 'var(--saffron)',
            fontSize: 11, fontWeight: 600, letterSpacing: 2,
            padding: '3px 14px', borderRadius: 20,
            textTransform: 'uppercase',
          }}>
            {result.hand_count === 2 ? 'Both hands' : '1 hand detected'}
          </div>
        </div>

        <div style={{ marginTop: 4 }}>
          <div style={{ display:'flex', justifyContent:'space-between',
                        marginBottom: 7, fontSize: 13 }}>
            <span style={{ color:'var(--text3)' }}>Confidence</span>
            <span style={{ fontWeight:700, color:'var(--text)' }}>
              {conf.toFixed(1)}%
            </span>
          </div>
          <div style={{ height: 7, background:'var(--border)',
                        borderRadius: 4, overflow:'hidden' }}>
            <div style={{
              height:'100%', borderRadius: 4,
              background: conf > 80
                ? 'linear-gradient(90deg, var(--saffron), var(--gold))'
                : conf > 50
                  ? 'linear-gradient(90deg, var(--gold), #d4a017)'
                  : '#ef4444',
              width: `${conf}%`,
              transition: 'width .8s ease',
            }}/>
          </div>
        </div>
      </Card>

      {/* Top 5 */}
      <Card title="Top Candidates">
        {top5.length === 0
          ? <p style={{ fontSize:13, color:'var(--text3)' }}>No data</p>
          : top5.map((p, i) => (
            <div key={i} style={{
              display:'flex', alignItems:'center', gap:10,
              padding:'8px 4px',
              borderBottom: i < top5.length-1
                ? '1px solid var(--border)' : 'none',
            }}>
              <span style={{ fontSize:11, color:'var(--text3)',
                             width:22, textAlign:'right', flexShrink:0 }}>
                #{i+1}
              </span>
              <div style={{ flex:1, minWidth:0 }}>
                <div style={{ fontSize:14, fontWeight:600,
                              color:'var(--text)', letterSpacing:.3 }}>
                  {p.name}
                </div>
              </div>
              <div style={{ width:72, flexShrink:0 }}>
                <div style={{ height:4, background:'var(--border)',
                              borderRadius:2 }}>
                  <div style={{
                    height:'100%', borderRadius:2,
                    background: i===0 ? 'var(--saffron)' : 'var(--border2)',
                    width:`${p.confidence_pct}%`, transition:'width .5s',
                  }}/>
                </div>
              </div>
              <span style={{ fontSize:12, color:'var(--text3)',
                             width:40, textAlign:'right', flexShrink:0 }}>
                {p.confidence_pct.toFixed(1)}%
              </span>
            </div>
          ))
        }
      </Card>

      {/* Detection info */}
      <Card title="Detection Info" accent="var(--teal)">
        {[
          ['Hands detected', result.hand_count ?? '—'],
          ['Hand labels', result.hand_labels?.join(', ') || '—'],
          ['Joints extracted', result.joints ? `${result.joints.length} joints` : '—'],
        ].map(([k, v]) => (
          <div key={k} style={{
            display:'flex', justifyContent:'space-between',
            fontSize:13, padding:'6px 0',
            borderBottom:'1px solid var(--border)',
          }}>
            <span style={{ color:'var(--text3)' }}>{k}</span>
            <span style={{ fontWeight:600, color:'var(--text)' }}>{v}</span>
          </div>
        ))}
      </Card>

    </div>
  );
}
