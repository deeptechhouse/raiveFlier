/**
 * ─── COMPOSITION DEFINITIONS ───
 *
 * Defines three Remotion compositions for the raiveFlier presentation:
 *
 *   1. raiveFlier-full   — Complete 18-slide presentation (153s at 30fps)
 *   2. raiveFlier-mini   — 7-slide RAG-focused mini presentation (67s at 30fps)
 *   3. raiveFlier-legacy — Original 10-slide presentation (90s at 30fps)
 *
 * Output resolution: 1920x1080 (Full HD) for all compositions.
 *
 * This is where Remotion discovers what videos can be rendered.
 * Each Presentation component handles its own slide sequencing internally.
 * The legacy composition is preserved for backward compatibility with
 * any existing rendering scripts that reference the old composition ID.
 */

import React from 'react';
import { Composition } from 'remotion';
import { PresentationFull } from './PresentationFull';
import { PresentationMini } from './PresentationMini';
import { Presentation } from './Presentation';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* Full 18-slide presentation — 4590 frames / 153 seconds */}
      <Composition
        id="raiveFlier-full"
        component={PresentationFull}
        durationInFrames={4590}
        fps={30}
        width={1920}
        height={1080}
      />

      {/* RAG-focused mini presentation — 2010 frames / 67 seconds */}
      <Composition
        id="raiveFlier-mini"
        component={PresentationMini}
        durationInFrames={2010}
        fps={30}
        width={1920}
        height={1080}
      />

      {/* Legacy 10-slide presentation — 2700 frames / 90 seconds */}
      <Composition
        id="raiveFlier-legacy"
        component={Presentation}
        durationInFrames={2700}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
