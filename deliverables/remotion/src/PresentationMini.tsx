/**
 * ─── RAG MINI PRESENTATION (7 SLIDES) ───
 *
 * A condensed 67-second composition focused on the RAG (Retrieval-Augmented
 * Generation) subsystem. This mini-presentation is designed for contexts
 * where only the RAG pipeline needs to be demonstrated — e.g., a focused
 * technical review or a component demo.
 *
 * Uses slides 1, 7, 8, 9, 10, 13, and 18 from the full presentation:
 *   - Title          (intro context)
 *   - RAG NonTech    (concept for non-technical audience)
 *   - RAG Tech       (implementation deep-dive)
 *   - Call Trace     (request flow showing RAG in context)
 *   - Code Examples  (interface + model code)
 *   - RAG Improvements (corpus expansion roadmap)
 *   - Outro          (closing)
 *
 * Total duration: 2010 frames (67 seconds at 30fps).
 *
 * Slide timing:
 *   Slide 1  Title              frames    0-239   (8s)
 *   Slide 2  RAG NonTech        frames  240-509   (9s)
 *   Slide 3  RAG Tech           frames  510-809   (10s)
 *   Slide 4  Call Trace         frames  810-1079  (9s)
 *   Slide 5  Code Examples      frames 1080-1349  (9s)
 *   Slide 6  RAG Improvements   frames 1350-1589  (8s)
 *   Slide 7  Outro              frames 1590-2009  (14s)
 *
 * Architecture connection: Composition orchestrator (subset).
 * Reuses the same slide components as PresentationFull — no
 * duplicate code, just different sequencing.
 *
 * Depends on: Subset of slide components, FilmGrain, theme.ts, Google Fonts.
 */

import React from 'react';
import { Sequence, AbsoluteFill } from 'remotion';
import { loadFont as loadSpaceGrotesk } from '@remotion/google-fonts/SpaceGrotesk';
import { loadFont as loadInter } from '@remotion/google-fonts/Inter';
import { loadFont as loadIBMPlexMono } from '@remotion/google-fonts/IBMPlexMono';

// Slide imports — only the RAG-relevant slides
import { TitleSlide } from './slides/TitleSlide';
import { RagNonTechSlide } from './slides/RagNonTechSlide';
import { RagTechSlide } from './slides/RagTechSlide';
import { CallTraceSlide } from './slides/CallTraceSlide';
import { CodeExamplesSlide } from './slides/CodeExamplesSlide';
import { RagImprovementsSlide } from './slides/RagImprovementsSlide';
import { OutroSlide } from './slides/OutroSlide';

// Overlay
import { FilmGrain } from './components/FilmGrain';
import { theme } from './theme';

// Load Google Fonts
loadSpaceGrotesk();
loadInter();
loadIBMPlexMono();

export const PresentationMini: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: theme.colors.bg }}>
      {/* Slide 1: Title card — establishes project context (8s) */}
      <Sequence from={0} durationInFrames={240}>
        <TitleSlide />
      </Sequence>

      {/* Slide 2: RAG explained for non-technical audience (9s) */}
      <Sequence from={240} durationInFrames={270}>
        <RagNonTechSlide />
      </Sequence>

      {/* Slide 3: RAG technical deep-dive with pipeline (10s) */}
      <Sequence from={510} durationInFrames={300}>
        <RagTechSlide />
      </Sequence>

      {/* Slide 4: Request flow showing RAG in the call chain (9s) */}
      <Sequence from={810} durationInFrames={270}>
        <CallTraceSlide />
      </Sequence>

      {/* Slide 5: Code examples — interface + model (9s) */}
      <Sequence from={1080} durationInFrames={270}>
        <CodeExamplesSlide />
      </Sequence>

      {/* Slide 6: Corpus expansion roadmap (8s) */}
      <Sequence from={1350} durationInFrames={240}>
        <RagImprovementsSlide />
      </Sequence>

      {/* Slide 7: Outro — closing with Q&A prompt (14s) */}
      <Sequence from={1590} durationInFrames={420}>
        <OutroSlide />
      </Sequence>

      {/* Film grain overlay — renders above all slides for texture */}
      <FilmGrain />
    </AbsoluteFill>
  );
};
