import React, { useState } from 'react';

const MUDRAS = [
  { name: 'Pataka',       meaning: 'Flag',              description: 'Four fingers straight and together, thumb bent. Represents a flag, forest, night, moonlight.' },
  { name: 'Tripataka',    meaning: 'Three Parts of Flag', description: 'Like Pataka but ring finger bent. Used for crown, tree, arrow, lightning.' },
  { name: 'Alapadma',     meaning: 'Full-Bloomed Lotus', description: 'All fingers spread. Represents a lotus in full bloom, beauty, a globe.' },
  { name: 'Mushti',       meaning: 'Fist',              description: 'All fingers closed in a fist. Represents firmness, stubbornness, fighting.' },
  { name: 'Shikhara',     meaning: 'Spire',             description: 'Thumb erect, fingers closed. Represents Shiva, a spire, a bow, love.' },
  { name: 'Katakamukha',  meaning: 'Opening of a Link', description: 'Index and middle finger together with thumb. Used for pearl strings, pulling a bowstring.' },
  { name: 'Mayura',       meaning: 'Peacock',           description: 'Thumb and index form a circle, others extended. Represents a peacock, eyebrow, neck of a bird.' },
  { name: 'Chandrakala',  meaning: 'Moon',              description: 'Modified Ardhachandra. Represents the moon, nectar, Shiva\'s forehead.' },
];

export default function AboutSection() {
  const [open, setOpen] = useState(false);

  return (
    <section style={{
      background: '#fff',
      borderBottom: '1px solid var(--border)',
      padding: '64px 48px',
    }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>

        {/* Section heading */}
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <div style={{
            width: 48, height: 3,
            background: 'var(--saffron)',
            margin: '0 auto 16px', borderRadius: 2,
          }}/>
          <h2 style={{
            fontFamily: 'Playfair Display', fontWeight: 700,
            fontSize: 'clamp(1.6rem, 3vw, 2.4rem)',
            color: 'var(--text)', marginBottom: 12,
          }}>
            What is Mudra in Bharatanatyam?
          </h2>
          <p style={{
            fontSize: 15, color: 'var(--text2)',
            maxWidth: 680, margin: '0 auto', lineHeight: 1.8,
          }}>
            In Bharatanatyam, <em>mudras</em> (literally "seal" or "sign") are
            codified hand gestures that communicate emotions, narratives, and
            symbolic meanings within classical Indian dance. They form the
            vocabulary of <em>abhinaya</em> — the art of expression.
          </p>
        </div>

        {/* Two-column cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: 24, marginBottom: 48,
        }}>
          {[
            {
              title: 'Asamyuta Mudras',
              subtitle: 'Single-hand gestures',
              text: 'Performed with one hand, these 28 mudras each carry distinct symbolic meaning. From Pataka (flag) representing the sky, to Mushti (fist) conveying determination, each finger position is precisely prescribed in the Natya Shastra — the ancient Sanskrit treatise on performing arts.',
              accent: 'var(--saffron)',
            },
            {
              title: 'Samyuta Mudras',
              subtitle: 'Both-hand gestures',
              text: 'Using both hands together, these 24 mudras create more complex meanings — depicting deities, natural phenomena, and abstract concepts. Our system detects both hands simultaneously to recognise these combined gestures.',
              accent: 'var(--maroon)',
            },
            {
              title: 'Natya Shastra',
              subtitle: 'The ancient treatise',
              text: 'Attributed to the sage Bharata Muni (200 BCE – 200 CE), the Natya Shastra codifies 108 karanas, 64 principles of dance, and precisely describes every hand gesture with its meaning and application in performance.',
              accent: 'var(--gold)',
            },
            {
              title: 'Symbolic Language',
              subtitle: 'Meaning through gesture',
              text: 'A single mudra can convey dozens of meanings depending on context: movement, eye expression, and body posture. Pataka alone can mean flag, wind, forest, waves, honour, night, river, or entering — context makes the difference.',
              accent: 'var(--teal)',
            },
          ].map(c => (
            <div key={c.title} style={{
              background: 'var(--warm)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              padding: '24px 22px',
              borderTop: `3px solid ${c.accent}`,
            }}>
              <div style={{ fontWeight: 600, fontSize: 16,
                            color: 'var(--text)', marginBottom: 4 }}>
                {c.title}
              </div>
              <div style={{ fontSize: 12, color: c.accent,
                            fontWeight: 600, letterSpacing: 1,
                            textTransform: 'uppercase', marginBottom: 12 }}>
                {c.subtitle}
              </div>
              <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.7 }}>
                {c.text}
              </p>
            </div>
          ))}
        </div>

        {/* Mudra quick reference */}
        <div style={{
          background: 'var(--warm)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          overflow: 'hidden',
        }}>
          <button
            onClick={() => setOpen(o => !o)}
            style={{
              width: '100%', background: 'none', border: 'none',
              padding: '16px 20px',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              cursor: 'pointer', fontFamily: 'Inter',
            }}
          >
            <span style={{ fontWeight: 600, fontSize: 15, color: 'var(--text)' }}>
              Common Bharatanatyam Mudras — Quick Reference
            </span>
            <span style={{
              fontSize: 20, color: 'var(--saffron)',
              transform: open ? 'rotate(180deg)' : 'none',
              transition: 'transform .25s',
              display: 'inline-block',
            }}>v</span>
          </button>

          {open && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
              gap: 1, borderTop: '1px solid var(--border)',
            }}>
              {MUDRAS.map(m => (
                <div key={m.name} style={{
                  padding: '14px 18px',
                  background: '#fff',
                  borderBottom: '1px solid var(--border)',
                }}>
                  <div style={{ fontWeight: 600, color: 'var(--text)',
                                fontSize: 14, marginBottom: 2 }}>
                    {m.name}
                    <span style={{ fontSize: 12, color: 'var(--saffron)',
                                   marginLeft: 8, fontWeight: 400 }}>
                      {m.meaning}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text3)',
                                lineHeight: 1.6 }}>
                    {m.description}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </section>
  );
}
