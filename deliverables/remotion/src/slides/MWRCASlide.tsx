/**
 * ─── MWRCA SLIDE ───
 *
 * Midwest Rave Culture Archive partnership slide (slide 17, frames 4140-4379).
 * Introduces the MWRCA collaboration — a curated archive of Midwest US
 * rave culture materials that raiveFlier can cross-reference for entity
 * enrichment and citation verification.
 *
 * Layout:
 *   - Title: "MWRCA Research Partnership"
 *   - Amber "Cuttable" badge indicating this slide can be cut from
 *     shorter presentations
 *   - Description paragraph
 *   - Four bullet points explaining the partnership's value
 *   - Footer with mwrca.org in mono font
 *
 * The "Cuttable" badge convention signals to the presenter that this
 * slide is supplementary and can be omitted for time-constrained talks.
 *
 * Architecture connection: Presentation layer. Describes an external
 * partnership that enriches the data layer (ChromaDB corpus).
 * Depends on: theme.ts, SlideTransition, Remotion spring.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { theme } from '../theme';

const partnershipPoints = [
  'Corpus enrichment from curated archives',
  'Cross-referencing entities with archival records',
  'Driving research traffic to mwrca.org',
  'Collaborative metadata verification',
];

export const MWRCASlide: React.FC = () => {
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
        {/* Title row with "Cuttable" badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 16,
            marginBottom: 16,
            opacity: titleOpacity,
          }}
        >
          <div
            style={{
              fontFamily: theme.fonts.display,
              fontSize: 42,
              fontWeight: 600,
              color: theme.colors.textPrimary,
            }}
          >
            MWRCA Research Partnership
          </div>

          {/* Cuttable badge — amber pill indicating this slide is optional */}
          <div
            style={{
              fontFamily: theme.fonts.mono,
              fontSize: 11,
              fontWeight: 600,
              color: theme.colors.amberText,
              backgroundColor: theme.colors.surfaceRaised,
              border: `1px solid ${theme.colors.amber}60`,
              padding: '4px 12px',
              borderRadius: 12,
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
            }}
          >
            Cuttable
          </div>
        </div>

        {/* Description */}
        <div
          style={{
            fontFamily: theme.fonts.body,
            fontSize: 18,
            color: theme.colors.textSecondary,
            maxWidth: 700,
            textAlign: 'center',
            lineHeight: 1.6,
            marginBottom: 40,
            opacity: spring({
              frame: Math.max(0, frame - 10),
              fps,
              config: { damping: 80, stiffness: 150 },
            }),
          }}
        >
          The Midwest Rave Culture Archive is a curated collection of
          materials documenting the electronic music scene across the
          Midwest United States. raiveFlier integrates with this archive
          for enriched entity research and citation verification.
        </div>

        {/* Partnership points */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 14,
            maxWidth: 600,
            width: '100%',
          }}
        >
          {partnershipPoints.map((point, index) => {
            const itemDelay = 20 + index * 8;
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
                  gap: 14,
                  opacity: itemOpacity,
                  transform: `translateX(${(1 - itemTranslateX) * -15}px)`,
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
                  {point}
                </span>
              </div>
            );
          })}
        </div>

        {/* Footer — mwrca.org reference */}
        <div
          style={{
            marginTop: 48,
            fontFamily: theme.fonts.mono,
            fontSize: 16,
            color: theme.colors.verdigrisText,
            letterSpacing: '0.06em',
            opacity: spring({
              frame: Math.max(0, frame - 60),
              fps,
              config: { damping: 80, stiffness: 150 },
            }),
          }}
        >
          mwrca.org
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
