/**
 * ─── SIMPLE LAYERED UML DIAGRAM ───
 *
 * Renders a stack of horizontal bars representing architecture layers,
 * each springing in from the top with staggered timing. This is a more
 * detailed variant of the ArchitectureSlide's pyramid — it includes
 * a "detail" subtitle on each layer bar for additional context.
 *
 * Used on the UMLPreviewSlide to show the dependency hierarchy with
 * brief descriptions of each layer's responsibility.
 *
 * Props:
 *   - layers: array of {name, detail, color} objects, ordered top-to-bottom
 *             (first element renders at the top of the stack)
 *
 * Animation strategy: Each layer springs in with 10-frame stagger from
 * index 0 downward, creating a cascading top-to-bottom reveal that
 * communicates the direction of dependency flow.
 *
 * Architecture connection: Presentational component used by UMLPreviewSlide.
 * Depends on: theme.ts for color/font tokens, Remotion for spring animation.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig } from 'remotion';
import { theme } from '../theme';

interface LayerDef {
  name: string;
  detail: string;
  color: string;
}

interface UMLDiagramProps {
  layers: LayerDef[];
}

export const UMLDiagram: React.FC<UMLDiagramProps> = ({ layers }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 6,
        width: '100%',
      }}
    >
      {layers.map((layer, index) => {
        // Each layer enters with staggered delay — top layer first
        const delay = index * 10;
        const opacity = spring({
          frame: Math.max(0, frame - delay),
          fps,
          config: { damping: 80, stiffness: 120 },
        });
        const translateY = spring({
          frame: Math.max(0, frame - delay),
          fps,
          config: { damping: 60, stiffness: 100, mass: 0.7 },
        });

        return (
          <div
            key={layer.name}
            style={{
              width: '100%',
              maxWidth: 700,
              height: 52,
              backgroundColor: theme.colors.surfaceRaised,
              border: `1px solid ${layer.color}`,
              borderRadius: 6,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 24px',
              opacity,
              transform: `translateY(${(1 - translateY) * -15}px)`,
            }}
          >
            {/* Layer name — left-aligned */}
            <span
              style={{
                fontFamily: theme.fonts.display,
                fontSize: 16,
                fontWeight: 600,
                color: layer.color,
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
              }}
            >
              {layer.name}
            </span>

            {/* Layer detail — right-aligned, muted */}
            <span
              style={{
                fontFamily: theme.fonts.mono,
                fontSize: 12,
                color: theme.colors.textMuted,
                letterSpacing: '0.04em',
              }}
            >
              {layer.detail}
            </span>
          </div>
        );
      })}
    </div>
  );
};
