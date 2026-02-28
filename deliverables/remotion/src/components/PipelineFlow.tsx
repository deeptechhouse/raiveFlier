/**
 * ─── HORIZONTAL PIPELINE FLOW ───
 *
 * Renders a horizontal sequence of labeled boxes connected by arrow
 * connectors, representing a data pipeline or processing flow.
 * Each step springs in with a staggered delay to create a left-to-right
 * cascade that visually communicates sequential processing.
 *
 * This component is more generic than the hard-coded pipeline in
 * RagTechSlide — it accepts arbitrary steps via props, making it
 * reusable across slides that need flow diagrams (FeederSlide,
 * UMLPreviewSlide, etc.).
 *
 * Props:
 *   - steps: array of {label, color} objects defining each pipeline box
 *
 * Animation strategy: Uses Remotion's spring() with stagger delay
 * (12 frames between steps). Each box fades in and scales up, then
 * the arrow to the next box fades in 8 frames later.
 *
 * Architecture connection: Reusable presentational component.
 * Depends on: theme.ts for color/font tokens, Remotion for animation.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig } from 'remotion';
import { theme } from '../theme';

interface PipelineStep {
  label: string;
  color: string;
}

interface PipelineFlowProps {
  steps: PipelineStep[];
}

export const PipelineFlow: React.FC<PipelineFlowProps> = ({ steps }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 0,
      }}
    >
      {steps.map((step, index) => {
        // Staggered entrance — each step delays 12 frames after the previous
        const stepDelay = index * 12;
        const stepOpacity = spring({
          frame: Math.max(0, frame - stepDelay),
          fps,
          config: { damping: 80, stiffness: 130 },
        });
        const stepScale = spring({
          frame: Math.max(0, frame - stepDelay),
          fps,
          config: { damping: 60, stiffness: 120, mass: 0.7 },
        });

        // Arrow appears 8 frames after its source step
        const arrowOpacity =
          index < steps.length - 1
            ? spring({
                frame: Math.max(0, frame - stepDelay - 8),
                fps,
                config: { damping: 80, stiffness: 150 },
              })
            : 0;

        return (
          <React.Fragment key={`${step.label}-${index}`}>
            {/* Step box */}
            <div
              style={{
                width: 120,
                height: 48,
                backgroundColor: theme.colors.surface,
                border: `1px solid ${step.color}`,
                borderRadius: 6,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                opacity: stepOpacity,
                transform: `scale(${stepScale})`,
              }}
            >
              <span
                style={{
                  fontFamily: theme.fonts.mono,
                  fontSize: 13,
                  fontWeight: 500,
                  color: step.color,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                }}
              >
                {step.label}
              </span>
            </div>

            {/* Arrow connector between steps */}
            {index < steps.length - 1 && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: 36,
                  opacity: arrowOpacity,
                }}
              >
                {/* Arrow shaft */}
                <div
                  style={{
                    width: 18,
                    height: 2,
                    backgroundColor: theme.colors.textMuted,
                  }}
                />
                {/* Arrow head */}
                <div
                  style={{
                    width: 0,
                    height: 0,
                    borderTop: '5px solid transparent',
                    borderBottom: '5px solid transparent',
                    borderLeft: `7px solid ${theme.colors.textMuted}`,
                  }}
                />
              </div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
