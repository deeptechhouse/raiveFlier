/**
 * ─── METRICS SLIDE ───
 *
 * Project statistics dashboard (frames 360-539, 6 seconds).
 * Displays six key metrics in a 3x2 grid using the MetricCounter component,
 * plus a progress bar showing overall project completeness.
 *
 * Each MetricCounter starts its count-up animation with a staggered delay
 * (8 frames apart) so the numbers cascade across the grid rather than
 * all starting simultaneously. This draws the viewer's eye across all stats.
 *
 * The progress bar at the bottom uses a spring animation to fill to ~92%,
 * visually communicating that the project is nearly feature-complete with
 * only a few roadmap items remaining.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { MetricCounter } from '../components/MetricCounter';
import { theme } from '../theme';

/** Metric definition with value, label, optional suffix, and stagger delay */
interface MetricDef {
  value: number;
  label: string;
  suffix?: string;
  delay: number;
}

const metrics: MetricDef[] = [
  { value: 30088, label: 'Lines of Python', delay: 0 },
  { value: 1174, label: 'Tests', delay: 8 },
  { value: 17, label: 'API Endpoints', delay: 16 },
  { value: 22, label: 'Provider Adapters', delay: 24 },
  { value: 9, label: 'Interfaces', delay: 32 },
  { value: 39, label: 'Pydantic Models', delay: 40 },
];

export const MetricsSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Progress bar fill — springs to 92% after a brief delay
  const barProgress = spring({
    frame: Math.max(0, frame - 50),
    fps,
    config: { damping: 60, stiffness: 80 },
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
            fontSize: 42,
            fontWeight: 600,
            color: theme.colors.textPrimary,
            marginBottom: 48,
          }}
        >
          By the Numbers
        </div>

        {/* 3x2 metrics grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '24px 60px',
            maxWidth: 900,
          }}
        >
          {metrics.map((m) => (
            <MetricCounter
              key={m.label}
              value={m.value}
              label={m.label}
              suffix={m.suffix}
              delay={m.delay}
            />
          ))}
        </div>

        {/* Progress bar — ~92% complete */}
        <div style={{ marginTop: 48, width: 600 }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontFamily: theme.fonts.mono,
              fontSize: 13,
              color: theme.colors.textMuted,
              marginBottom: 8,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
            }}
          >
            <span>Project Completeness</span>
            <span>{Math.round(92 * barProgress)}%</span>
          </div>
          <div
            style={{
              height: 8,
              backgroundColor: theme.colors.surfaceRaised,
              borderRadius: 4,
              overflow: 'hidden',
            }}
          >
            {/* Filled portion — amethyst gradient */}
            <div
              style={{
                height: '100%',
                width: `${92 * barProgress}%`,
                background: `linear-gradient(90deg, ${theme.colors.amethyst}, ${theme.colors.amethystHover})`,
                borderRadius: 4,
                transition: 'none',
              }}
            />
          </div>
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
