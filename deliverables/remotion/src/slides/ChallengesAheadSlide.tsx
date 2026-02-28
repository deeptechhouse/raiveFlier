/**
 * ─── CHALLENGES AHEAD SLIDE ───
 *
 * Forward-looking roadmap slide (slide 15, frames 3660-3899).
 * Complements the "Problems Solved" ChallengesSlide by focusing
 * exclusively on upcoming work — the "Road Ahead" items that were
 * split out from the original combined challenges slide to give
 * each topic adequate screen time.
 *
 * Five roadmap items are presented with staggered spring entrance,
 * each with a verdigris bullet to maintain visual consistency with
 * the "Road Ahead" column from the legacy ChallengesSlide.
 *
 * Architecture connection: Presentation layer. Complements ChallengesSlide.
 * Depends on: theme.ts, SlideTransition, Remotion spring.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { theme } from '../theme';

const roadAhead = [
  'Production scaling \u2014 multi-worker, Redis, CDN',
  'Audio fingerprinting \u2014 Shazam-style event matching',
  'Geospatial timeline \u2014 map visualization of rave history',
  'Community contributions \u2014 user-submitted fliers',
  'Expanded corpus \u2014 RA Exchange, interviews, academic sources',
];

export const ChallengesAheadSlide: React.FC = () => {
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
            fontSize: 42,
            fontWeight: 600,
            color: theme.colors.textPrimary,
            marginBottom: 48,
            opacity: titleOpacity,
          }}
        >
          Road Ahead
        </div>

        {/* Roadmap items */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 18,
            maxWidth: 700,
            width: '100%',
          }}
        >
          {roadAhead.map((item, index) => {
            const itemDelay = 10 + index * 8;
            const itemOpacity = spring({
              frame: Math.max(0, frame - itemDelay),
              fps,
              config: { damping: 80, stiffness: 150 },
            });
            const itemTranslateX = spring({
              frame: Math.max(0, frame - itemDelay),
              fps,
              config: { damping: 60, stiffness: 120, mass: 0.6 },
            });

            return (
              <div
                key={index}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 16,
                  backgroundColor: theme.colors.surfaceRaised,
                  padding: '16px 24px',
                  borderRadius: 8,
                  borderLeft: `3px solid ${theme.colors.verdigris}`,
                  opacity: itemOpacity,
                  transform: `translateX(${(1 - itemTranslateX) * -20}px)`,
                }}
              >
                {/* Verdigris bullet */}
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor: theme.colors.verdigris,
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontFamily: theme.fonts.body,
                    fontSize: 17,
                    color: theme.colors.textSecondary,
                    lineHeight: 1.5,
                  }}
                >
                  {item}
                </span>
              </div>
            );
          })}
        </div>

        {/* Footer note */}
        <div
          style={{
            marginTop: 48,
            fontFamily: theme.fonts.mono,
            fontSize: 14,
            color: theme.colors.textMuted,
            opacity: spring({
              frame: Math.max(0, frame - 60),
              fps,
              config: { damping: 80, stiffness: 150 },
            }),
            letterSpacing: '0.04em',
          }}
        >
          Contributions welcome at github.com/deeptechhouse/raiveFlier
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
