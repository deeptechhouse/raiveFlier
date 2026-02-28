/**
 * ─── FULL PRESENTATION SEQUENCE (18 SLIDES) ───
 *
 * Orchestrates all 18 slides using Remotion's <Sequence> component.
 * This is the primary composition for the complete raiveFlier presentation,
 * covering the full arc from introduction through features, architecture,
 * implementation, roadmap, and closing.
 *
 * Total duration: 4590 frames (153 seconds at 30fps).
 *
 * Slide timing breakdown:
 *   Slide  1  Title              frames     0-239   ( 8s)
 *   Slide  2  Features Overview  frames   240-509   ( 9s)
 *   Slide  3  Walkthrough        frames   510-779   ( 9s)
 *   Slide  4  Metrics            frames   780-1019  ( 8s)
 *   Slide  5  Completeness       frames  1020-1259  ( 8s)
 *   Slide  6  Architecture       frames  1260-1559  (10s)
 *   Slide  7  RAG NonTech        frames  1560-1829  ( 9s)
 *   Slide  8  RAG Tech           frames  1830-2129  (10s)
 *   Slide  9  Call Trace         frames  2130-2399  ( 9s)
 *   Slide 10  Code Examples      frames  2400-2669  ( 9s)
 *   Slide 11  UML Preview        frames  2670-2909  ( 8s)
 *   Slide 12  Feeder             frames  2910-3179  ( 9s)
 *   Slide 13  RAG Improvements   frames  3180-3419  ( 8s)
 *   Slide 14  Challenges Solved  frames  3420-3659  ( 8s)
 *   Slide 15  Challenges Ahead   frames  3660-3899  ( 8s)
 *   Slide 16  Git History        frames  3900-4139  ( 8s)
 *   Slide 17  MWRCA              frames  4140-4379  ( 8s)
 *   Slide 18  Outro              frames  4380-4589  ( 7s)
 *                                         ----
 *                                TOTAL: 4590 frames / 153s
 *
 * The FilmGrain overlay sits above all slides to match the raiveFlier
 * app's visual texture. Google Fonts are loaded at this top level
 * so all children inherit them.
 *
 * Architecture connection: Composition orchestrator. Imports all slide
 * components and arranges them sequentially. Each slide is self-contained
 * and receives its own local frame counter via Remotion's <Sequence>.
 *
 * Depends on: All slide components, FilmGrain, theme.ts, Google Fonts.
 */

import React from 'react';
import { Sequence, AbsoluteFill } from 'remotion';
import { loadFont as loadSpaceGrotesk } from '@remotion/google-fonts/SpaceGrotesk';
import { loadFont as loadInter } from '@remotion/google-fonts/Inter';
import { loadFont as loadIBMPlexMono } from '@remotion/google-fonts/IBMPlexMono';

// Slide imports — ordered by presentation sequence
import { TitleSlide } from './slides/TitleSlide';
import { FeaturesOverviewSlide } from './slides/FeaturesOverviewSlide';
import { WalkthroughSlide } from './slides/WalkthroughSlide';
import { MetricsSlide } from './slides/MetricsSlide';
import { CompletenessSlide } from './slides/CompletenessSlide';
import { ArchitectureSlide } from './slides/ArchitectureSlide';
import { RagNonTechSlide } from './slides/RagNonTechSlide';
import { RagTechSlide } from './slides/RagTechSlide';
import { CallTraceSlide } from './slides/CallTraceSlide';
import { CodeExamplesSlide } from './slides/CodeExamplesSlide';
import { UMLPreviewSlide } from './slides/UMLPreviewSlide';
import { FeederSlide } from './slides/FeederSlide';
import { RagImprovementsSlide } from './slides/RagImprovementsSlide';
import { ChallengesSlide } from './slides/ChallengesSlide';
import { ChallengesAheadSlide } from './slides/ChallengesAheadSlide';
import { GitHistorySlide } from './slides/GitHistorySlide';
import { MWRCASlide } from './slides/MWRCASlide';
import { OutroSlide } from './slides/OutroSlide';

// Overlay
import { FilmGrain } from './components/FilmGrain';
import { theme } from './theme';

// Load Google Fonts so they are available during rendering.
// Each loadFont call returns font-face CSS that Remotion injects.
loadSpaceGrotesk();
loadInter();
loadIBMPlexMono();

export const PresentationFull: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: theme.colors.bg }}>
      {/* Slide 1: Title card (8s) */}
      <Sequence from={0} durationInFrames={240}>
        <TitleSlide />
      </Sequence>

      {/* Slide 2: Features overview — 6 capability cards (9s) */}
      <Sequence from={240} durationInFrames={270}>
        <FeaturesOverviewSlide />
      </Sequence>

      {/* Slide 3: User walkthrough — 4-step flow (9s) */}
      <Sequence from={510} durationInFrames={270}>
        <WalkthroughSlide />
      </Sequence>

      {/* Slide 4: Project metrics — animated counters (8s) */}
      <Sequence from={780} durationInFrames={240}>
        <MetricsSlide />
      </Sequence>

      {/* Slide 5: Completeness — ~92% progress (8s) */}
      <Sequence from={1020} durationInFrames={240}>
        <CompletenessSlide />
      </Sequence>

      {/* Slide 6: Layered architecture diagram (10s) */}
      <Sequence from={1260} durationInFrames={300}>
        <ArchitectureSlide />
      </Sequence>

      {/* Slide 7: RAG explained for non-technical audience (9s) */}
      <Sequence from={1560} durationInFrames={270}>
        <RagNonTechSlide />
      </Sequence>

      {/* Slide 8: RAG technical deep-dive with pipeline (10s) */}
      <Sequence from={1830} durationInFrames={300}>
        <RagTechSlide />
      </Sequence>

      {/* Slide 9: Call trace / request flow diagram (9s) */}
      <Sequence from={2130} durationInFrames={270}>
        <CallTraceSlide />
      </Sequence>

      {/* Slide 10: Code examples — interface + model (9s) */}
      <Sequence from={2400} durationInFrames={270}>
        <CodeExamplesSlide />
      </Sequence>

      {/* Slide 11: UML architecture detail (8s) */}
      <Sequence from={2670} durationInFrames={240}>
        <UMLPreviewSlide />
      </Sequence>

      {/* Slide 12: raiveFeeder companion tool (9s) */}
      <Sequence from={2910} durationInFrames={270}>
        <FeederSlide />
      </Sequence>

      {/* Slide 13: Corpus expansion roadmap (8s) */}
      <Sequence from={3180} durationInFrames={240}>
        <RagImprovementsSlide />
      </Sequence>

      {/* Slide 14: Problems solved (8s) */}
      <Sequence from={3420} durationInFrames={240}>
        <ChallengesSlide />
      </Sequence>

      {/* Slide 15: Road ahead (8s) */}
      <Sequence from={3660} durationInFrames={240}>
        <ChallengesAheadSlide />
      </Sequence>

      {/* Slide 16: Development velocity — git stats (8s) */}
      <Sequence from={3900} durationInFrames={240}>
        <GitHistorySlide />
      </Sequence>

      {/* Slide 17: MWRCA research partnership (8s) */}
      <Sequence from={4140} durationInFrames={240}>
        <MWRCASlide />
      </Sequence>

      {/* Slide 18: Outro — quote, wordmark, Q&A (7s) */}
      <Sequence from={4380} durationInFrames={210}>
        <OutroSlide />
      </Sequence>

      {/* Film grain overlay — renders above all slides for texture */}
      <FilmGrain />
    </AbsoluteFill>
  );
};
