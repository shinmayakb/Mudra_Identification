import React from 'react';

export default function HeroSection() {
  return (
    <section style={{
      background: 'linear-gradient(135deg, #FFF8F0 0%, #FEF3E8 50%, #F9EDD8 100%)',
      borderBottom: '1px solid var(--border)',
      padding: '80px 48px',
      textAlign: 'center',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Decorative rangoli rings */}
      {[200, 340, 480].map((size, i) => (
        <div key={i} style={{
          position: 'absolute',
          top: '50%', left: '50%',
          width: size, height: size,
          borderRadius: '50%',
          border: `1px solid rgba(232,114,12,${0.06 - i * 0.015})`,
          transform: 'translate(-50%, -50%)',
          pointerEvents: 'none',
        }}/>
      ))}
      {/* Decorative dots */}
      {Array.from({length: 8}).map((_, i) => {
        const a = (i / 8) * Math.PI * 2;
        const r = 160;
        return (
          <div key={i} style={{
            position: 'absolute',
            top: `calc(50% + ${Math.sin(a) * r}px)`,
            left: `calc(50% + ${Math.cos(a) * r}px)`,
            width: 6, height: 6, borderRadius: '50%',
            background: 'rgba(232,114,12,0.25)',
            transform: 'translate(-50%,-50%)',
            pointerEvents: 'none',
          }}/>
        );
      })}

      <div style={{ position: 'relative', maxWidth: 720, margin: '0 auto' }}>
        <div style={{
          display: 'inline-block',
          background: 'rgba(232,114,12,.1)',
          border: '1px solid rgba(232,114,12,.3)',
          color: 'var(--saffron)',
          padding: '5px 18px', borderRadius: 40,
          fontSize: 11, fontWeight: 600, letterSpacing: 3,
          textTransform: 'uppercase', marginBottom: 22,
        }}>
          IEEE TMM 2025 — VIE + JWC Architecture
        </div>

        <h1 style={{
          fontFamily: 'Playfair Display', fontWeight: 700,
          fontSize: 'clamp(2rem, 5vw, 3.4rem)',
          color: 'var(--text)', lineHeight: 1.2,
          marginBottom: 16,
        }}>
          Bharatanatyam Mudra
          <br/>
          <span style={{ color: 'var(--saffron)' }}>Recognition System</span>
        </h1>

        <p style={{
          fontSize: 16, color: 'var(--text2)', lineHeight: 1.8,
          maxWidth: 600, margin: '0 auto 32px',
        }}>
          Recognise classical Indian hand gestures using Graph Convolutional Networks
          with Viewpoint Influence Elimination and Joint-Weighted Classification —
          works from any angle with single or dual hands.
        </p>

        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
          {[
            ['JGCN + KGCN', 'Graph Convolutional Networks'],
            ['VIE', 'Viewpoint Invariance'],
            ['JWC', 'Joint-Weighted Classification'],
            ['Dual Hand', 'Single & Both Hands'],
          ].map(([label, desc]) => (
            <div key={label} style={{
              background: '#fff',
              border: '1px solid var(--border)',
              borderRadius: 10, padding: '10px 16px',
              textAlign: 'center',
              boxShadow: '0 1px 4px rgba(0,0,0,.06)',
            }}>
              <div style={{ fontSize: 13, fontWeight: 600,
                            color: 'var(--saffron)' }}>{label}</div>
              <div style={{ fontSize: 11, color: 'var(--text3)',
                            marginTop: 2 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
