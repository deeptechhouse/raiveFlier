/**
 * ─── UML BALL-AND-SOCKET CONNECTOR ───
 *
 * Renders a simplified UML "ball-and-socket" interface symbol used in
 * component diagrams to show that one component provides an interface
 * (ball/lollipop) and another component requires it (socket/arc).
 *
 * In raiveFlier's architecture, every external service is accessed
 * through an abstract interface (ILLMProvider, IOCRProvider, etc.).
 * This visual connector reinforces that design principle on the
 * UML preview slide.
 *
 * Visual structure:
 *   - Ball: filled circle in amethyst — represents the "provided" interface
 *   - Socket: half-arc in verdigris — represents the "required" interface
 *   - Label: interface name displayed below the connector
 *
 * Props:
 *   - label:     interface name (e.g., "ILLMProvider")
 *   - direction: which side the socket faces ('left' | 'right', default 'right')
 *
 * Architecture connection: Presentational component used on UMLPreviewSlide.
 * Depends on: theme.ts for color/font tokens.
 */

import React from 'react';
import { theme } from '../theme';

interface BallSocketConnectorProps {
  label: string;
  direction?: 'left' | 'right';
}

export const BallSocketConnector: React.FC<BallSocketConnectorProps> = ({
  label,
  direction = 'right',
}) => {
  // Flip the layout based on direction so the socket faces the correct side
  const isRight = direction === 'right';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 8,
      }}
    >
      {/* Connector graphic — ball + socket aligned horizontally */}
      <div
        style={{
          display: 'flex',
          flexDirection: isRight ? 'row' : 'row-reverse',
          alignItems: 'center',
          gap: 0,
        }}
      >
        {/* Ball (provided interface) — filled circle */}
        <div
          style={{
            width: 16,
            height: 16,
            borderRadius: '50%',
            backgroundColor: theme.colors.amethyst,
          }}
        />

        {/* Connecting line between ball and socket */}
        <div
          style={{
            width: 12,
            height: 2,
            backgroundColor: theme.colors.textMuted,
          }}
        />

        {/* Socket (required interface) — half-arc using border-radius */}
        <div
          style={{
            width: 16,
            height: 24,
            borderTop: `2px solid ${theme.colors.verdigris}`,
            borderBottom: `2px solid ${theme.colors.verdigris}`,
            ...(isRight
              ? {
                  borderRight: `2px solid ${theme.colors.verdigris}`,
                  borderTopRightRadius: 12,
                  borderBottomRightRadius: 12,
                }
              : {
                  borderLeft: `2px solid ${theme.colors.verdigris}`,
                  borderTopLeftRadius: 12,
                  borderBottomLeftRadius: 12,
                }),
          }}
        />
      </div>

      {/* Interface label */}
      <span
        style={{
          fontFamily: theme.fonts.mono,
          fontSize: 11,
          color: theme.colors.textMuted,
          letterSpacing: '0.04em',
        }}
      >
        {label}
      </span>
    </div>
  );
};
