/**
 * ─── CODE EXAMPLES SLIDE ───
 *
 * Side-by-side code display (frames 1410-1649, 8 seconds).
 * Shows two key code excerpts that demonstrate raiveFlier's design:
 *
 *   Left:  ILLMProvider interface — demonstrates the adapter pattern
 *          that enables vendor-agnostic LLM integration
 *   Right: PipelineState model — demonstrates immutable state management
 *          using Pydantic's frozen models
 *
 * Both panels use the CodeBlock component's typewriter animation,
 * with the right panel starting 20 frames after the left to create
 * a staggered reveal that guides the viewer's eye.
 */

import React from 'react';
import { useCurrentFrame, spring, useVideoConfig, AbsoluteFill } from 'remotion';
import { SlideTransition } from '../components/SlideTransition';
import { CodeBlock } from '../components/CodeBlock';
import { theme } from '../theme';
import { interfaceCode, pipelineStateCode } from '../data/codeSnippets';

export const CodeExamplesSlide: React.FC = () => {
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
            fontSize: 38,
            fontWeight: 600,
            color: theme.colors.textPrimary,
            marginBottom: 40,
            opacity: titleOpacity,
          }}
        >
          Design Patterns in Practice
        </div>

        {/* Two code panels side by side */}
        <div
          style={{
            display: 'flex',
            gap: 32,
            width: '100%',
            maxWidth: 1600,
          }}
        >
          {/* Left panel: interface — adapter pattern contract */}
          <div style={{ flex: 1 }}>
            <CodeBlock
              code={interfaceCode}
              title="src/interfaces/llm_provider.py"
              delay={10}
            />
          </div>

          {/* Right panel: model — immutable pipeline state */}
          <div style={{ flex: 1 }}>
            <CodeBlock
              code={pipelineStateCode}
              title="src/models/pipeline.py"
              delay={30}
            />
          </div>
        </div>

        {/* Annotation below the code */}
        <div
          style={{
            display: 'flex',
            gap: 32,
            width: '100%',
            maxWidth: 1600,
            marginTop: 20,
          }}
        >
          <div
            style={{
              flex: 1,
              fontFamily: theme.fonts.mono,
              fontSize: 13,
              color: theme.colors.amethystText,
              textAlign: 'center',
              opacity: spring({
                frame: Math.max(0, frame - 80),
                fps,
                config: { damping: 80, stiffness: 150 },
              }),
            }}
          >
            Adapter Pattern — swap providers without touching business logic
          </div>
          <div
            style={{
              flex: 1,
              fontFamily: theme.fonts.mono,
              fontSize: 13,
              color: theme.colors.verdigrisText,
              textAlign: 'center',
              opacity: spring({
                frame: Math.max(0, frame - 100),
                fps,
                config: { damping: 80, stiffness: 150 },
              }),
            }}
          >
            Immutable State — transitions via copy, never mutation
          </div>
        </div>
      </AbsoluteFill>
    </SlideTransition>
  );
};
