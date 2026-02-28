/**
 * ─── CHALLENGES SLIDE ───
 *
 * Two-column retrospective (frames 1650-1799, 5 seconds).
 * Left column: "Problems Solved" (amethyst accent) — key technical
 * challenges that were overcome during the 6-day build sprint.
 * Right column: "Road Ahead" (verdigris accent) — planned improvements
 * and future work items.
 *
 * List items enter with staggered spring animations within each column,
 * and the right column starts slightly after the left to guide the
 * viewer's reading flow from accomplishments to future plans.
 *
 * A footer mentions mwrca.org as the project's context/community.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { theme } from '../theme';

const problemsSolved = [
  'OCR on stylized rave typography (90s fonts, neon, distortion)',
  '512MB Docker RAM \u2014 excluded PyTorch, used FastEmbed ONNX',
  'Multi-database entity deduplication across 5 sources',
  'Citation authority ranking (3-tier verification system)',
  'Immutable pipeline state \u2014 Pydantic frozen models',
];

const roadAhead = [
  'Production scaling',
  'Audio fingerprinting',
  'Geospatial timeline',
  'Community contributions',
  'Expanded corpus sources',
];

export const ChallengesSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

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
          Challenges & Road Ahead
        </div>

        {/* Two-column layout */}
        <div
          style={{
            display: 'flex',
            gap: 80,
            width: '100%',
            maxWidth: 1200,
          }}
        >
          {/* Left column: Problems Solved */}
          <div style={{ flex: 1 }}>
            <div
              style={{
                fontFamily: theme.fonts.display,
                fontSize: 22,
                fontWeight: 600,
                color: theme.colors.amethystText,
                marginBottom: 24,
                opacity: titleOpacity,
              }}
            >
              Problems Solved
            </div>
            {problemsSolved.map((item, index) => {
              const itemOpacity = spring({
                frame: Math.max(0, frame - 10 - index * 6),
                fps,
                config: { damping: 80, stiffness: 150 },
              });
              const itemTranslateX = spring({
                frame: Math.max(0, frame - 10 - index * 6),
                fps,
                config: { damping: 60, stiffness: 120, mass: 0.6 },
              });

              return (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 12,
                    marginBottom: 14,
                    opacity: itemOpacity,
                    transform: `translateX(${(1 - itemTranslateX) * -15}px)`,
                  }}
                >
                  {/* Amethyst bullet */}
                  <div
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      backgroundColor: theme.colors.amethyst,
                      marginTop: 8,
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontFamily: theme.fonts.body,
                      fontSize: 16,
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

          {/* Right column: Road Ahead */}
          <div style={{ flex: 1 }}>
            <div
              style={{
                fontFamily: theme.fonts.display,
                fontSize: 22,
                fontWeight: 600,
                color: theme.colors.verdigrisText,
                marginBottom: 24,
                opacity: titleOpacity,
              }}
            >
              Road Ahead
            </div>
            {roadAhead.map((item, index) => {
              // Right column starts 20 frames after left
              const itemOpacity = spring({
                frame: Math.max(0, frame - 30 - index * 6),
                fps,
                config: { damping: 80, stiffness: 150 },
              });
              const itemTranslateX = spring({
                frame: Math.max(0, frame - 30 - index * 6),
                fps,
                config: { damping: 60, stiffness: 120, mass: 0.6 },
              });

              return (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 12,
                    marginBottom: 14,
                    opacity: itemOpacity,
                    transform: `translateX(${(1 - itemTranslateX) * -15}px)`,
                  }}
                >
                  {/* Verdigris bullet */}
                  <div
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      backgroundColor: theme.colors.verdigris,
                      marginTop: 8,
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontFamily: theme.fonts.body,
                      fontSize: 16,
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
        </div>

        {/* Footer — community context */}
        <div
          style={{
            marginTop: 40,
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
          Built for the community at mwrca.org
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
