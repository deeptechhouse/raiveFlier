/**
 * ─── TITLE SLIDE ───
 *
 * Opening card for the raiveFlier presentation (frames 0-149, 5 seconds).
 * Displays the project name in large Space Grotesk display text,
 * a descriptive subtitle, the "Upload a rave flier. Get back the history."
 * tagline, and an MIT license badge.
 *
 * Each text element enters with a staggered spring animation — the title
 * arrives first, subtitle 10 frames later, tagline 20 frames after that,
 * and the badge last. This creates a cascading reveal effect.
 *
 * A subtle amethyst glow line sits below the title to add visual weight
 * and tie into the accent color system.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { theme } from '../theme';

export const TitleSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Staggered springs for cascading text entrance
  const titleOpacity = spring({ frame, fps, config: { damping: 80, stiffness: 150 } });
  const subtitleOpacity = spring({ frame: Math.max(0, frame - 10), fps, config: { damping: 80, stiffness: 150 } });
  const taglineOpacity = spring({ frame: Math.max(0, frame - 20), fps, config: { damping: 80, stiffness: 150 } });
  const badgeOpacity = spring({ frame: Math.max(0, frame - 35), fps, config: { damping: 80, stiffness: 150 } });

  // Title slides upward as it fades in
  const titleTranslate = spring({ frame, fps, config: { damping: 80, stiffness: 150, mass: 0.8 } });

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
        {/* Project name — large display type */}
        <div
          style={{
            fontFamily: theme.fonts.display,
            fontSize: 96,
            fontWeight: 700,
            color: theme.colors.textPrimary,
            opacity: titleOpacity,
            transform: `translateY(${(1 - titleTranslate) * 40}px)`,
            letterSpacing: '-0.02em',
          }}
        >
          raiveFlier
        </div>

        {/* Amethyst accent line beneath the title */}
        <div
          style={{
            width: 120,
            height: 3,
            background: `linear-gradient(90deg, transparent, ${theme.colors.amethyst}, transparent)`,
            marginTop: 16,
            marginBottom: 32,
            opacity: titleOpacity,
          }}
        />

        {/* Subtitle — describes what the project is */}
        <div
          style={{
            fontFamily: theme.fonts.body,
            fontSize: 28,
            color: theme.colors.textSecondary,
            opacity: subtitleOpacity,
            textAlign: 'center',
            maxWidth: 800,
          }}
        >
          An open-source analysis engine for electronic music culture
        </div>

        {/* Tagline — the elevator pitch */}
        <div
          style={{
            fontFamily: theme.fonts.mono,
            fontSize: 20,
            color: theme.colors.amethystText,
            opacity: taglineOpacity,
            marginTop: 24,
            letterSpacing: '0.04em',
          }}
        >
          Upload a rave flier. Get back the history.
        </div>

        {/* MIT license badge */}
        <div
          style={{
            marginTop: 48,
            opacity: badgeOpacity,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <div
            style={{
              fontFamily: theme.fonts.mono,
              fontSize: 13,
              color: theme.colors.textMuted,
              padding: '6px 14px',
              border: `1px solid ${theme.colors.border}`,
              borderRadius: 4,
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
            }}
          >
            MIT License
          </div>
          <div
            style={{
              fontFamily: theme.fonts.mono,
              fontSize: 13,
              color: theme.colors.textMuted,
              padding: '6px 14px',
              border: `1px solid ${theme.colors.border}`,
              borderRadius: 4,
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
            }}
          >
            Python 3.12
          </div>
          <div
            style={{
              fontFamily: theme.fonts.mono,
              fontSize: 13,
              color: theme.colors.textMuted,
              padding: '6px 14px',
              border: `1px solid ${theme.colors.border}`,
              borderRadius: 4,
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
            }}
          >
            FastAPI
          </div>
          {/* Completeness badge — amethyst pill showing current progress */}
          <div
            style={{
              fontFamily: theme.fonts.mono,
              fontSize: 13,
              color: theme.colors.amethystText,
              padding: '6px 14px',
              border: `1px solid ${theme.colors.amethyst}`,
              borderRadius: 4,
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
            }}
          >
            ~92% Complete
          </div>
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
