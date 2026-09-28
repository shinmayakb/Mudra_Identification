import React from 'react';

export default function Header({ modelReady, numClasses, onTrain }) {
  return (
    <header style={{
      background: '#fff',
      borderBottom: '1px solid var(--border)',
      padding: '0 48px',
      height: 64,
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      position: 'sticky', top: 0, zIndex: 100,
      boxShadow: '0 1px 4px rgba(0,0,0,.06)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{
          width: 36, height: 36,
          background: 'linear-gradient(135deg, var(--saffron), var(--gold))',
          borderRadius: 8,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
               stroke="white" strokeWidth="2" strokeLinecap="round">
            <path d="M18 10V6a2 2 0 00-4 0v4"/>
            <path d="M14 6V4a2 2 0 00-4 0v6"/>
            <path d="M10 5V4a2 2 0 00-4 0v10"/>
            <path d="M6 14v-3a2 2 0 014 0"/>
            <rect x="2" y="14" width="20" height="8" rx="2"/>
          </svg>
        </div>
        <div>
          <div style={{ fontFamily: 'Playfair Display', fontWeight: 700,
                        fontSize: 16, color: 'var(--text)', letterSpacing: .3 }}>
            Mudra Recognition
          </div>
          <div style={{ fontSize: 10, color: 'var(--text3)', letterSpacing: 2,
                        textTransform: 'uppercase' }}>
            Bharatanatyam
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6,
                      fontSize: 13, color: 'var(--text2)' }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: modelReady ? 'var(--ok)' : 'var(--err)',
            boxShadow: modelReady ? '0 0 6px #15803d80' : '0 0 6px #b91c1c80',
          }}/>
          {modelReady ? `Model ready (${numClasses} classes)` : 'Model not trained'}
        </div>
        <button onClick={onTrain} style={{
          background: 'var(--saffron)',
          border: 'none', color: '#fff',
          padding: '8px 20px', borderRadius: 7,
          fontFamily: 'Inter', fontSize: 13, fontWeight: 600,
          cursor: 'pointer', letterSpacing: .3,
          transition: 'background .2s',
        }}
          onMouseEnter={e => e.target.style.background='var(--saf-dk)'}
          onMouseLeave={e => e.target.style.background='var(--saffron)'}
        >
          Train Model
        </button>
      </div>
    </header>
  );
}
