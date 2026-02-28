/**
 * ─── GIT HISTORY SLIDE ───
 *
 * Development velocity dashboard (slide 16, frames 3900-4139).
 * Showcases the rapid development pace of raiveFlier:
 *   - "145 commits in 9 days" as large animated counters
 *   - "23 Pull Requests" as an additional counter
 *   - A simplified bar chart approximating daily commit frequency
 *   - Workflow badges (MIT License, GitHub Flow, Annotated Code)
 *
 * The bar chart uses 9 bars (one per day) with heights based on an
 * approximate distribution averaging ~16 commits/day. Each bar springs
 * upward with staggered timing for a wave effect.
 *
 * Architecture connection: Presentation layer. Pulls metric values from
 * src/data/metrics.ts for consistency with other slides.
 * Depends on: theme.ts, SlideTransition, MetricCounter, Remotion spring.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { MetricCounter } from '../components/MetricCounter';
import { theme } from '../theme';
import { metrics } from '../data/metrics';

/**
 * Approximate daily commit distribution across 9 development days.
 * Total sums to 145 (matching metrics.commits).
 * Distribution reflects typical sprint patterns: ramp-up, peak, taper.
 */
const dailyCommits = [8, 12, 18, 22, 20, 19, 17, 16, 13];

/** Workflow badges displayed at the bottom of the slide */
const badges = ['MIT License', 'GitHub Flow', 'Annotated Code'];

export const GitHistorySlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title entrance
  const titleOpacity = spring({
    frame,
    fps,
    config: { damping: 80, stiffness: 150 },
  });

  // Maximum daily commits — used to normalize bar heights
  const maxCommits = Math.max(...dailyCommits);

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
            marginBottom: 32,
            opacity: titleOpacity,
          }}
        >
          Development Velocity
        </div>

        {/* Large metric counters row */}
        <div
          style={{
            display: 'flex',
            gap: 80,
            marginBottom: 40,
          }}
        >
          <MetricCounter
            value={metrics.commits}
            label={`Commits in ${metrics.devDays} Days`}
            delay={5}
          />
          <MetricCounter
            value={metrics.pullRequests}
            label="Pull Requests"
            delay={15}
          />
        </div>

        {/* Bar chart — daily commit distribution */}
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            gap: 12,
            height: 120,
            marginBottom: 8,
          }}
        >
          {dailyCommits.map((count, index) => {
            // Each bar springs up with staggered delay
            const barDelay = 25 + index * 6;
            const barProgress = spring({
              frame: Math.max(0, frame - barDelay),
              fps,
              config: { damping: 60, stiffness: 100 },
            });

            // Normalize height: tallest bar = 100px, others proportional
            const barHeight = (count / maxCommits) * 100 * barProgress;

            return (
              <div
                key={index}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                {/* Bar */}
                <div
                  style={{
                    width: 40,
                    height: barHeight,
                    background: `linear-gradient(180deg, ${theme.colors.amethystHover}, ${theme.colors.amethyst})`,
                    borderRadius: '4px 4px 0 0',
                  }}
                />
                {/* Day label */}
                <span
                  style={{
                    fontFamily: theme.fonts.mono,
                    fontSize: 10,
                    color: theme.colors.textMuted,
                  }}
                >
                  D{index + 1}
                </span>
              </div>
            );
          })}
        </div>

        {/* Chart label */}
        <div
          style={{
            fontFamily: theme.fonts.mono,
            fontSize: 12,
            color: theme.colors.textMuted,
            marginBottom: 32,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}
        >
          Commits per Day (approximate)
        </div>

        {/* Workflow badges */}
        <div
          style={{
            display: 'flex',
            gap: 12,
            opacity: spring({
              frame: Math.max(0, frame - 70),
              fps,
              config: { damping: 80, stiffness: 150 },
            }),
          }}
        >
          {badges.map((badge) => (
            <div
              key={badge}
              style={{
                fontFamily: theme.fonts.mono,
                fontSize: 12,
                color: theme.colors.textMuted,
                padding: '6px 14px',
                border: `1px solid ${theme.colors.border}`,
                borderRadius: 4,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
              }}
            >
              {badge}
            </div>
          ))}
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
