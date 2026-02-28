/**
 * ─── FEEDER SLIDE ───
 *
 * Introduces raiveFeeder, the companion corpus management tool (slide 12,
 * frames 2910-3179). raiveFeeder runs on port 8001 alongside the main
 * raiveFlier app and provides a dedicated interface for ingesting documents,
 * audio, images, and URLs into the shared ChromaDB vector store.
 *
 * Layout:
 *   - Title: "raiveFeeder — Corpus Companion"
 *   - 5 horizontal pill tabs representing content types
 *   - Connection diagram: raiveFlier <-> ChromaDB <-> raiveFeeder
 *   - Port badge in mono font
 *
 * Each tab springs in with staggered timing, and the connection diagram
 * fades in after the tabs to show how the two apps share a data layer.
 *
 * Architecture connection: Presentation layer. Describes the relationship
 * between raiveFlier and raiveFeeder, both of which use ChromaDB as the
 * shared vector store. raiveFeeder is a FastAPI sub-application mounted
 * at /feeder on port 8001.
 *
 * Depends on: theme.ts, SlideTransition, Remotion spring.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { theme } from '../theme';

/** Tab definition for content types raiveFeeder supports */
interface FeederTab {
  label: string;
  description: string;
}

const tabs: FeederTab[] = [
  { label: 'Documents', description: 'PDFs, text files, markdown' },
  { label: 'Audio', description: 'Transcripts from interviews' },
  { label: 'Images', description: 'Flier scans, venue photos' },
  { label: 'URLs', description: 'Web pages, articles, blogs' },
  { label: 'Corpus', description: 'Browse and manage chunks' },
];

/** Connection diagram nodes */
const connectionNodes = ['raiveFlier', 'ChromaDB', 'raiveFeeder'];

export const FeederSlide: React.FC = () => {
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
        {/* Title */}
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
          raiveFeeder
        </div>

        {/* Subtitle */}
        <div
          style={{
            fontFamily: theme.fonts.body,
            fontSize: 20,
            color: theme.colors.textSecondary,
            marginBottom: 40,
            opacity: titleOpacity,
          }}
        >
          Corpus Companion
        </div>

        {/* Horizontal pill tabs for content types */}
        <div
          style={{
            display: 'flex',
            gap: 12,
            marginBottom: 32,
          }}
        >
          {tabs.map((tab, index) => {
            const tabDelay = 10 + index * 8;
            const tabOpacity = spring({
              frame: Math.max(0, frame - tabDelay),
              fps,
              config: { damping: 80, stiffness: 150 },
            });

            return (
              <div
                key={tab.label}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 8,
                  opacity: tabOpacity,
                }}
              >
                {/* Pill button */}
                <div
                  style={{
                    fontFamily: theme.fonts.mono,
                    fontSize: 13,
                    fontWeight: 500,
                    color:
                      index === 0
                        ? theme.colors.bg
                        : theme.colors.textSecondary,
                    backgroundColor:
                      index === 0
                        ? theme.colors.amethystText
                        : theme.colors.surfaceRaised,
                    padding: '8px 18px',
                    borderRadius: 20,
                    border: `1px solid ${
                      index === 0
                        ? theme.colors.amethystText
                        : theme.colors.border
                    }`,
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                  }}
                >
                  {tab.label}
                </div>

                {/* Tab description */}
                <span
                  style={{
                    fontFamily: theme.fonts.body,
                    fontSize: 12,
                    color: theme.colors.textMuted,
                    textAlign: 'center',
                    maxWidth: 120,
                  }}
                >
                  {tab.description}
                </span>
              </div>
            );
          })}
        </div>

        {/* Connection diagram: raiveFlier <-> ChromaDB <-> raiveFeeder */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 0,
            marginTop: 16,
            opacity: spring({
              frame: Math.max(0, frame - 60),
              fps,
              config: { damping: 80, stiffness: 120 },
            }),
          }}
        >
          {connectionNodes.map((node, index) => {
            // Node colors: amethyst for raiveFlier, amber for ChromaDB, verdigris for raiveFeeder
            const nodeColor =
              index === 0
                ? theme.colors.amethyst
                : index === 1
                  ? theme.colors.amber
                  : theme.colors.verdigris;

            return (
              <React.Fragment key={node}>
                {/* Node box */}
                <div
                  style={{
                    width: 180,
                    height: 52,
                    backgroundColor: theme.colors.surface,
                    border: `2px solid ${nodeColor}`,
                    borderRadius: 8,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <span
                    style={{
                      fontFamily: theme.fonts.display,
                      fontSize: 16,
                      fontWeight: 500,
                      color: nodeColor,
                    }}
                  >
                    {node}
                  </span>
                </div>

                {/* Bidirectional arrow between nodes */}
                {index < connectionNodes.length - 1 && (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: 60,
                      color: theme.colors.textMuted,
                      fontFamily: theme.fonts.mono,
                      fontSize: 20,
                    }}
                  >
                    {'\u2194'}
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Port badge */}
        <div
          style={{
            marginTop: 32,
            fontFamily: theme.fonts.mono,
            fontSize: 14,
            color: theme.colors.textMuted,
            padding: '6px 16px',
            border: `1px solid ${theme.colors.border}`,
            borderRadius: 4,
            letterSpacing: '0.08em',
            opacity: spring({
              frame: Math.max(0, frame - 80),
              fps,
              config: { damping: 80, stiffness: 150 },
            }),
          }}
        >
          PORT 8001
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
