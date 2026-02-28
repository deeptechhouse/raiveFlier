/**
 * ─── UML PREVIEW SLIDE ───
 *
 * Simplified layered architecture diagram (slide 11, frames 2670-2909).
 * Uses the UMLDiagram component to render a detailed version of the
 * architecture stack, with each layer showing both its name and a
 * brief description of its responsibility.
 *
 * This complements the earlier ArchitectureSlide (which shows a tapered
 * pyramid) by providing more detail about what each layer actually does
 * — bridging the conceptual overview with the implementation specifics
 * shown in subsequent code/trace slides.
 *
 * The BallSocketConnector components at the bottom illustrate the
 * interface contracts that connect adjacent layers, reinforcing
 * raiveFlier's adapter pattern / dependency inversion architecture.
 *
 * Architecture connection: Presentation layer.
 * Depends on: theme.ts, SlideTransition, UMLDiagram, BallSocketConnector.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { UMLDiagram } from '../components/UMLDiagram';
import { BallSocketConnector } from '../components/BallSocketConnector';
import { theme } from '../theme';

/**
 * Architecture layers with detailed descriptions, ordered top-to-bottom.
 * Colors match the ArchitectureSlide palette for visual consistency.
 */
const architectureLayers = [
  { name: 'Frontend', detail: 'Vanilla JS/CSS/HTML', color: theme.colors.verdigris },
  { name: 'API', detail: 'FastAPI routes + schemas', color: theme.colors.amethyst },
  { name: 'Pipeline', detail: 'Orchestrator + confirmation gate', color: theme.colors.amber },
  { name: 'Services', detail: 'Research, citation, Q&A, recs', color: theme.colors.verdigris },
  { name: 'Interfaces', detail: '9 abstract base classes', color: theme.colors.amethystHover },
  { name: 'Providers', detail: '22 concrete adapters', color: theme.colors.amethyst },
  { name: 'Models', detail: '39 Pydantic v2 frozen objects', color: theme.colors.amethystMuted },
];

/** Key interfaces to display as ball-and-socket connectors */
const keyInterfaces = ['ILLMProvider', 'IOCRProvider', 'IVectorStore'];

export const UMLPreviewSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title entrance
  const titleOpacity = spring({
    frame,
    fps,
    config: { damping: 80, stiffness: 150 },
  });

  return (
    <SlideTransition>
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.slidePadding,
        }}
      >
        {/* Section title */}
        <div
          style={{
            fontFamily: theme.fonts.display,
            fontSize: 38,
            fontWeight: 600,
            color: theme.colors.textPrimary,
            marginBottom: 32,
            opacity: titleOpacity,
          }}
        >
          Architecture Detail
        </div>

        {/* Layered UML diagram */}
        <div style={{ width: '100%', maxWidth: 750, marginBottom: 32 }}>
          <UMLDiagram layers={architectureLayers} />
        </div>

        {/* Interface connectors — ball-and-socket symbols */}
        <div
          style={{
            display: 'flex',
            gap: 48,
            opacity: spring({
              frame: Math.max(0, frame - 80),
              fps,
              config: { damping: 80, stiffness: 120 },
            }),
          }}
        >
          {keyInterfaces.map((iface) => (
            <BallSocketConnector key={iface} label={iface} />
          ))}
        </div>

        {/* Architecture principle footer */}
        <div
          style={{
            marginTop: 24,
            fontFamily: theme.fonts.mono,
            fontSize: 13,
            color: theme.colors.textMuted,
            letterSpacing: '0.06em',
            opacity: spring({
              frame: Math.max(0, frame - 90),
              fps,
              config: { damping: 80, stiffness: 150 },
            }),
          }}
        >
          Every external service abstracted behind an interface — zero vendor lock-in
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
