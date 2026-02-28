/**
 * ─── COMPLETENESS SLIDE ───
 *
 * Project completion status dashboard (slide 5, frames 1020-1259).
 * Displays a large animated "~92%" headline, an amethyst gradient
 * progress bar, and two columns listing completed features vs.
 * remaining work items.
 *
 * The large percentage uses MetricCounter's spring animation to count
 * up from 0 to 92, creating visual drama. The progress bar fills with
 * a spring-driven width transition that lands at 92%.
 *
 * Two-column layout:
 *   Left:  "Done" — checkmark-prefixed items in success green
 *   Right: "Remaining" — circle-prefixed items in muted text
 *
 * Architecture connection: Presentation layer. Imports metrics from
 * src/data/metrics.ts for the completeness percentage.
 * Depends on: theme.ts, SlideTransition, Remotion spring.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { theme } from '../theme';
import { metrics } from '../data/metrics';

const doneItems = [
  'OCR pipeline (multi-engine)',
  'Entity extraction',
  'Multi-DB research (5 sources)',
  'RAG corpus + vector search',
  'Interconnection discovery',
  'Q&A engine with citations',
  'Recommendations engine',
  'Frontend (vanilla JS)',
  'API layer (17 endpoints)',
  'Docker deployment',
];

const remainingItems = [
  'Production scaling',
  'Audio fingerprinting',
  'Geospatial timeline',
  'Community pipeline',
];

export const CompletenessSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Large percentage counter — springs from 0 to completeness value
  const counterProgress = spring({
    frame: Math.max(0, frame - 5),
    fps,
    config: { damping: 50, stiffness: 80 },
  });

  // Progress bar fill — springs to completeness percentage after brief delay
  const barProgress = spring({
    frame: Math.max(0, frame - 20),
    fps,
    config: { damping: 60, stiffness: 80 },
  });

  // Title entrance
  const titleOpacity = spring({
    frame,
    fps,
    config: { damping: 80, stiffness: 150 },
  });

  const displayPercent = Math.round(metrics.completeness * counterProgress);

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
            marginBottom: 24,
            opacity: titleOpacity,
          }}
        >
          Project Completeness
        </div>

        {/* Large animated percentage */}
        <div
          style={{
            fontFamily: theme.fonts.display,
            fontSize: 96,
            fontWeight: 700,
            color: theme.colors.amethystText,
            marginBottom: 16,
          }}
        >
          ~{displayPercent}%
        </div>

        {/* Progress bar */}
        <div style={{ width: 600, marginBottom: 40 }}>
          <div
            style={{
              height: 10,
              backgroundColor: theme.colors.surfaceRaised,
              borderRadius: 5,
              overflow: 'hidden',
            }}
          >
            {/* Filled portion — amethyst gradient */}
            <div
              style={{
                height: '100%',
                width: `${metrics.completeness * barProgress}%`,
                background: `linear-gradient(90deg, ${theme.colors.amethyst}, ${theme.colors.amethystHover})`,
                borderRadius: 5,
              }}
            />
          </div>
        </div>

        {/* Two-column Done / Remaining */}
        <div
          style={{
            display: 'flex',
            gap: 80,
            width: '100%',
            maxWidth: 1000,
          }}
        >
          {/* Done column */}
          <div style={{ flex: 1 }}>
            <div
              style={{
                fontFamily: theme.fonts.display,
                fontSize: 20,
                fontWeight: 600,
                color: theme.colors.success,
                marginBottom: 16,
                opacity: titleOpacity,
              }}
            >
              Done
            </div>
            {doneItems.map((item, index) => {
              const itemOpacity = spring({
                frame: Math.max(0, frame - 30 - index * 4),
                fps,
                config: { damping: 80, stiffness: 150 },
              });

              return (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    marginBottom: 8,
                    opacity: itemOpacity,
                  }}
                >
                  {/* Checkmark icon */}
                  <span
                    style={{
                      fontFamily: theme.fonts.mono,
                      fontSize: 14,
                      color: theme.colors.success,
                      flexShrink: 0,
                    }}
                  >
                    {'\u2713'}
                  </span>
                  <span
                    style={{
                      fontFamily: theme.fonts.body,
                      fontSize: 14,
                      color: theme.colors.textSecondary,
                      lineHeight: 1.4,
                    }}
                  >
                    {item}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Remaining column */}
          <div style={{ flex: 1 }}>
            <div
              style={{
                fontFamily: theme.fonts.display,
                fontSize: 20,
                fontWeight: 600,
                color: theme.colors.textMuted,
                marginBottom: 16,
                opacity: titleOpacity,
              }}
            >
              Remaining
            </div>
            {remainingItems.map((item, index) => {
              const itemOpacity = spring({
                frame: Math.max(0, frame - 50 - index * 6),
                fps,
                config: { damping: 80, stiffness: 150 },
              });

              return (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    marginBottom: 10,
                    opacity: itemOpacity,
                  }}
                >
                  {/* Open circle icon */}
                  <div
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      border: `1.5px solid ${theme.colors.textMuted}`,
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontFamily: theme.fonts.body,
                      fontSize: 14,
                      color: theme.colors.textMuted,
                      lineHeight: 1.4,
                    }}
                  >
                    {item}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
