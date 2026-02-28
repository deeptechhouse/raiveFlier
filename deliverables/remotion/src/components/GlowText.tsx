/**
 * ─── ANIMATED GLOW TEXT ───
 *
 * Reusable text component that renders with an amethyst-tinted glow effect
 * and optional breathing pulse animation. Used on emphasis text across
 * multiple slides (OutroSlide quote, FeederSlide title, etc.) to draw
 * visual attention without being overpowering.
 *
 * The glow is achieved via CSS text-shadow with two layered blur radii —
 * a tight 20px glow for definition and a soft 60px glow for atmosphere.
 * The pulse uses a sine wave on the frame counter to create a gentle
 * breathing oscillation between 85% and 100% opacity.
 *
 * Props:
 *   - text:     the string to render
 *   - fontSize: font size in px (default 48)
 *   - color:    text color (default amethystText from theme)
 *   - glow:     whether to apply the pulsing glow (default true)
 *
 * Architecture connection: Presentational component used by slides.
 * Depends on: theme.ts for color/font tokens, Remotion for frame access.
 */

import React from 'react';
import { useCurrentFrame } from 'remotion';
import { theme } from '../theme';

interface GlowTextProps {
  text: string;
  fontSize?: number;
  color?: string;
  glow?: boolean;
}

export const GlowText: React.FC<GlowTextProps> = ({
  text,
  fontSize = 48,
  color = theme.colors.amethystText,
  glow = true,
}) => {
  const frame = useCurrentFrame();

  // Gentle sine-wave pulse — oscillates opacity between 0.85 and 1.0.
  // The 0.04 multiplier keeps the frequency slow (one full cycle ~157 frames / ~5.2s at 30fps).
  const pulse = glow ? 0.85 + 0.15 * Math.sin(frame * 0.04) : 1;

  // Two-layer text-shadow: tight glow + diffuse atmosphere
  const glowShadow = glow
    ? `0 0 20px ${theme.colors.amethyst}80, 0 0 60px ${theme.colors.amethyst}40`
    : 'none';

  return (
    <div
      style={{
        fontFamily: theme.fonts.display,
        fontSize,
        fontWeight: 700,
        color,
        opacity: pulse,
        textShadow: glowShadow,
        textAlign: 'center',
        lineHeight: 1.3,
      }}
    >
      {text}
    </div>
  );
};
