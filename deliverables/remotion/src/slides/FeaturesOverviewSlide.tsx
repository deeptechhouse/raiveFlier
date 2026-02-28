/**
 * ─── FEATURES OVERVIEW SLIDE ───
 *
 * Six feature cards in a 2x3 grid showcasing raiveFlier's core capabilities.
 * This slide provides a high-level feature map early in the presentation
 * (slide 2, frames 240-509) so the audience understands the project's
 * breadth before diving into architecture and implementation details.
 *
 * Each card has:
 *   - A colored left border (amethyst, verdigris, or amber rotating)
 *   - A title in display font
 *   - A brief description in body font
 *
 * Cards stagger in with spring animations (8-frame delay between each),
 * reading left-to-right, top-to-bottom — matching natural reading order.
 *
 * Color rotation pattern:
 *   Row 1: amethyst, verdigris, amber
 *   Row 2: amethyst, verdigris, amber
 * This creates visual rhythm without being monotonous.
 *
 * Architecture connection: Presentation layer. No business logic.
 * Depends on: theme.ts, SlideTransition, Remotion spring.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { theme } from '../theme';

/** Feature card definition */
interface Feature {
  title: string;
  description: string;
  color: string;
}

const features: Feature[] = [
  {
    title: 'OCR Pipeline',
    description: 'Multi-engine OCR with LLM vision fallback',
    color: theme.colors.amethyst,
  },
  {
    title: 'Multi-DB Research',
    description: '5 music databases queried in parallel',
    color: theme.colors.verdigris,
  },
  {
    title: 'RAG Corpus',
    description: 'Vector search with 3-tier citations',
    color: theme.colors.amber,
  },
  {
    title: 'Interconnection',
    description: 'Entity relationship discovery',
    color: theme.colors.amethyst,
  },
  {
    title: 'Q&A Engine',
    description: 'NL answers with inline citations',
    color: theme.colors.verdigris,
  },
  {
    title: 'Recommendations',
    description: 'Related artist/event discovery',
    color: theme.colors.amber,
  },
];

export const FeaturesOverviewSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title entrance animation
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
          What raiveFlier Does
        </div>

        {/* 2x3 feature card grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 24,
            maxWidth: 1100,
            width: '100%',
          }}
        >
          {features.map((feature, index) => {
            // Staggered entrance — 8 frames between each card
            const cardDelay = 10 + index * 8;
            const cardOpacity = spring({
              frame: Math.max(0, frame - cardDelay),
              fps,
              config: { damping: 80, stiffness: 150 },
            });
            const cardTranslateY = spring({
              frame: Math.max(0, frame - cardDelay),
              fps,
              config: { damping: 60, stiffness: 120, mass: 0.6 },
            });

            return (
              <div
                key={feature.title}
                style={{
                  backgroundColor: theme.colors.surfaceRaised,
                  borderRadius: 8,
                  padding: '24px 20px',
                  borderLeft: `3px solid ${feature.color}`,
                  opacity: cardOpacity,
                  transform: `translateY(${(1 - cardTranslateY) * 20}px)`,
                }}
              >
                {/* Feature title */}
                <div
                  style={{
                    fontFamily: theme.fonts.display,
                    fontSize: 20,
                    fontWeight: 600,
                    color: theme.colors.textPrimary,
                    marginBottom: 8,
                  }}
                >
                  {feature.title}
                </div>

                {/* Feature description */}
                <div
                  style={{
                    fontFamily: theme.fonts.body,
                    fontSize: 15,
                    color: theme.colors.textSecondary,
                    lineHeight: 1.5,
                  }}
                >
                  {feature.description}
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
