/**
 * ─── OUTRO SLIDE ───
 *
 * Closing slide for the presentation (slide 18, frames 4380-4589).
 * Combines an evocative quote about rave flier culture, the project
 * wordmark, link badges for project resources, a tech attribution
 * line, and a "Questions?" prompt with a blinking cursor.
 *
 * The quote uses a subtle amethyst glow via text-shadow to add
 * visual weight and emotional resonance. The blinking cursor
 * (toggling every 30 frames = 1s) signals the end of the automated
 * presentation and the start of live Q&A.
 *
 * Layout (top to bottom):
 *   1. Large italicized quote
 *   2. raiveFlier wordmark in Space Grotesk
 *   3. Three link badges: GitHub, Live Demo, Docs
 *   4. Tech attribution line
 *   5. "Questions?" with blinking cursor
 *
 * Architecture connection: Presentation layer — final slide.
 * Depends on: theme.ts, SlideTransition, Remotion spring.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { theme } from '../theme';

/** Resource link badges for the footer */
const linkBadges = ['GitHub', 'Live Demo', 'Docs'];

export const OutroSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Staggered entrances for each section
  const quoteOpacity = spring({
    frame,
    fps,
    config: { damping: 80, stiffness: 150 },
  });
  const wordmarkOpacity = spring({
    frame: Math.max(0, frame - 15),
    fps,
    config: { damping: 80, stiffness: 150 },
  });
  const badgesOpacity = spring({
    frame: Math.max(0, frame - 30),
    fps,
    config: { damping: 80, stiffness: 150 },
  });
  const attributionOpacity = spring({
    frame: Math.max(0, frame - 45),
    fps,
    config: { damping: 80, stiffness: 150 },
  });
  const questionsOpacity = spring({
    frame: Math.max(0, frame - 60),
    fps,
    config: { damping: 80, stiffness: 150 },
  });

  // Blinking cursor — toggles visibility every 30 frames (1s at 30fps)
  const cursorVisible = Math.floor(frame / 30) % 2 === 0;

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
        {/* Evocative quote — italicized with amethyst glow */}
        <div
          style={{
            fontFamily: theme.fonts.body,
            fontSize: 28,
            fontStyle: 'italic',
            color: theme.colors.textPrimary,
            textAlign: 'center',
            maxWidth: 900,
            lineHeight: 1.5,
            opacity: quoteOpacity,
            textShadow: `0 0 30px ${theme.colors.amethyst}40`,
            marginBottom: 40,
          }}
        >
          &ldquo;The fliers were the internet before the internet.&rdquo;
        </div>

        {/* raiveFlier wordmark */}
        <div
          style={{
            fontFamily: theme.fonts.display,
            fontSize: 72,
            fontWeight: 700,
            color: theme.colors.textPrimary,
            letterSpacing: '-0.02em',
            opacity: wordmarkOpacity,
            marginBottom: 32,
          }}
        >
          raiveFlier
        </div>

        {/* Link badges: GitHub, Live Demo, Docs */}
        <div
          style={{
            display: 'flex',
            gap: 12,
            marginBottom: 32,
            opacity: badgesOpacity,
          }}
        >
          {linkBadges.map((badge) => (
            <div
              key={badge}
              style={{
                fontFamily: theme.fonts.mono,
                fontSize: 13,
                color: theme.colors.amethystText,
                padding: '8px 20px',
                border: `1px solid ${theme.colors.amethyst}`,
                borderRadius: 6,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
              }}
            >
              {badge}
            </div>
          ))}
        </div>

        {/* Tech attribution */}
        <div
          style={{
            fontFamily: theme.fonts.mono,
            fontSize: 14,
            color: theme.colors.textMuted,
            textAlign: 'center',
            lineHeight: 1.5,
            opacity: attributionOpacity,
            marginBottom: 40,
            letterSpacing: '0.04em',
          }}
        >
          Built with Python, FastAPI, ChromaDB, and respect for the culture
        </div>

        {/* "Questions?" prompt with blinking cursor */}
        <div
          style={{
            fontFamily: theme.fonts.display,
            fontSize: 36,
            fontWeight: 600,
            color: theme.colors.amethystText,
            opacity: questionsOpacity,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          Questions?
          <span
            style={{
              opacity: cursorVisible ? 1 : 0,
              color: theme.colors.amethystText,
              fontWeight: 400,
            }}
          >
            |
          </span>
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
