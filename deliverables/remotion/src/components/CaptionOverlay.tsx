/**
 * ─── CAPTION OVERLAY BAR ───
 *
 * A fixed-position caption bar that sits at the bottom of the slide,
 * displaying supplementary text in monospace font. Designed to provide
 * contextual notes, source references, or slide numbers without
 * interfering with the main slide content.
 *
 * The bar uses a semi-transparent dark background (rgba(10,10,12,0.85))
 * that blends with the Bunker palette while remaining readable against
 * any slide content above it.
 *
 * Props:
 *   - text: caption string to display
 *
 * Architecture connection: Presentational overlay component.
 * Designed to be placed inside a slide's AbsoluteFill alongside
 * the main content — it positions itself at the bottom via
 * absolute positioning within the fill container.
 */

import React from 'react';
import { theme } from '../theme';

interface CaptionOverlayProps {
  text: string;
}

export const CaptionOverlay: React.FC<CaptionOverlayProps> = ({ text }) => {
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: 60,
        backgroundColor: 'rgba(10, 10, 12, 0.85)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <span
        style={{
          fontFamily: theme.fonts.mono,
          fontSize: 16,
          color: theme.colors.textSecondary,
          letterSpacing: '0.04em',
        }}
      >
        {text}
      </span>
    </div>
  );
};
