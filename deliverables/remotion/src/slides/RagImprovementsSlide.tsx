/**
 * ─── RAG IMPROVEMENTS SLIDE ───
 *
 * Corpus expansion roadmap (slide 13, frames 3180-3419).
 * Lists planned sources for enriching the raiveFlier reference corpus,
 * each with a status indicator showing they are in the planning phase.
 *
 * This slide communicates the project's forward vision for corpus growth
 * — moving beyond the initial reference_corpus/ seed data to include
 * curated cultural archives, interview transcripts, and academic sources.
 *
 * Each item springs in with staggered timing (8-frame delay) and has
 * a status pill badge showing "Planned" to indicate these are roadmap
 * items, not yet implemented features.
 *
 * Architecture connection: Presentation layer. Describes the planned
 * evolution of the data layer (ChromaDB corpus ingestion pipeline).
 * Depends on: theme.ts, SlideTransition, Remotion spring.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { theme } from '../theme';

/** Planned corpus source with description */
interface CorpusSource {
  name: string;
  description: string;
}

const plannedSources: CorpusSource[] = [
  {
    name: 'RA Exchange interviews',
    description: 'Long-form DJ and producer conversations from Resident Advisor',
  },
  {
    name: 'Resident Advisor event listings',
    description: 'Historical event data for venue and promoter context',
  },
  {
    name: 'Interview transcripts / oral histories',
    description: 'First-person accounts from scene participants and documentarians',
  },
  {
    name: 'Academic papers / music studies',
    description: 'Peer-reviewed research on electronic music culture and history',
  },
  {
    name: 'Community-submitted fliers',
    description: 'User-contributed scans expanding geographic and temporal coverage',
  },
];

export const RagImprovementsSlide: React.FC = () => {
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
            marginBottom: 12,
            opacity: titleOpacity,
          }}
        >
          Corpus Expansion Roadmap
        </div>

        {/* Subtitle */}
        <div
          style={{
            fontFamily: theme.fonts.mono,
            fontSize: 15,
            color: theme.colors.textMuted,
            marginBottom: 48,
            opacity: titleOpacity,
            letterSpacing: '0.06em',
          }}
        >
          Planned Reference Sources
        </div>

        {/* Source list */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
            maxWidth: 800,
            width: '100%',
          }}
        >
          {plannedSources.map((source, index) => {
            const itemDelay = 15 + index * 8;
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
                key={source.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 16,
                  backgroundColor: theme.colors.surfaceRaised,
                  borderRadius: 8,
                  padding: '16px 20px',
                  border: `1px solid ${theme.colors.border}`,
                  opacity: itemOpacity,
                  transform: `translateX(${(1 - itemTranslateX) * -20}px)`,
                }}
              >
                {/* Status pill */}
                <div
                  style={{
                    fontFamily: theme.fonts.mono,
                    fontSize: 10,
                    fontWeight: 600,
                    color: theme.colors.amberText,
                    backgroundColor: theme.colors.surfaceRaised,
                    border: `1px solid ${theme.colors.amber}40`,
                    padding: '4px 10px',
                    borderRadius: 10,
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em',
                    flexShrink: 0,
                  }}
                >
                  Planned
                </div>

                {/* Source info */}
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      fontFamily: theme.fonts.display,
                      fontSize: 17,
                      fontWeight: 600,
                      color: theme.colors.textPrimary,
                      marginBottom: 4,
                    }}
                  >
                    {source.name}
                  </div>
                  <div
                    style={{
                      fontFamily: theme.fonts.body,
                      fontSize: 14,
                      color: theme.colors.textSecondary,
                      lineHeight: 1.4,
                    }}
                  >
                    {source.description}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
