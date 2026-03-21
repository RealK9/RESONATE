/**
 * RESONATE — Animated Orb.
 * Organic swirling shape inspired by the RESONATE logo.
 * Pulsates and flows during audio analysis with brand gradient colors.
 * Uses SVG paths with CSS animations for a fluid, living feel.
 */

import { useEffect, useRef } from "react";
import { useTheme } from "../theme/ThemeProvider";

// Generate organic blob path using sinusoidal perturbation
function blobPath(cx, cy, baseR, points, phase, wobble) {
  const pts = [];
  for (let i = 0; i <= points; i++) {
    const angle = (i / points) * Math.PI * 2;
    const r = baseR +
      Math.sin(angle * 3 + phase) * wobble * 0.6 +
      Math.sin(angle * 5 + phase * 1.4) * wobble * 0.3 +
      Math.cos(angle * 2 + phase * 0.7) * wobble * 0.4;
    pts.push([cx + Math.cos(angle) * r, cy + Math.sin(angle) * r]);
  }
  // Smooth closed path using quadratic curves
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const curr = pts[i];
    const next = pts[i + 1];
    const mx = (curr[0] + next[0]) / 2;
    const my = (curr[1] + next[1]) / 2;
    d += ` Q ${curr[0]} ${curr[1]} ${mx} ${my}`;
  }
  d += " Z";
  return d;
}

export function ResonateOrb({ progress = 0, size = 260 }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const { mode } = useTheme();
  const isDark = mode === "dark";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2;
    const baseR = size * 0.28;
    let startTime = Date.now();

    function draw() {
      const t = (Date.now() - startTime) / 1000;
      const p = progress / 100;
      ctx.clearRect(0, 0, size, size);

      // Number of flowing strands
      const strandCount = 12;

      for (let s = 0; s < strandCount; s++) {
        const strandPhase = (s / strandCount) * Math.PI * 2;
        const speed = 0.4 + s * 0.08;
        const phase = t * speed + strandPhase;
        const wobble = 12 + p * 18 + Math.sin(t * 0.5 + s) * 6;
        const r = baseR - 8 + s * 2.5;

        // Build path
        const path = new Path2D(blobPath(cx, cy, r, 64, phase, wobble));

        // Create gradient for each strand
        const gradAngle = t * 0.3 + strandPhase;
        const gx1 = cx + Math.cos(gradAngle) * size * 0.4;
        const gy1 = cy + Math.sin(gradAngle) * size * 0.4;
        const gx2 = cx - Math.cos(gradAngle) * size * 0.4;
        const gy2 = cy - Math.sin(gradAngle) * size * 0.4;

        const grad = ctx.createLinearGradient(gx1, gy1, gx2, gy2);

        // Brand colors: pink → purple → cyan
        const alpha = (0.06 + (s / strandCount) * 0.12) * (0.6 + p * 0.4);
        grad.addColorStop(0, `rgba(217, 70, 239, ${alpha})`);
        grad.addColorStop(0.4, `rgba(139, 92, 246, ${alpha * 0.9})`);
        grad.addColorStop(1, `rgba(6, 182, 212, ${alpha})`);

        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.2 + p * 0.8;
        ctx.stroke(path);
      }

      // Bright inner glow strands (fewer, brighter)
      for (let s = 0; s < 5; s++) {
        const strandPhase = (s / 5) * Math.PI * 2;
        const phase = t * (0.6 + s * 0.12) + strandPhase + Math.PI;
        const wobble = 8 + p * 22 + Math.sin(t * 0.7 + s * 1.3) * 8;
        const r = baseR - 4 + s * 3;

        const path = new Path2D(blobPath(cx, cy, r, 64, phase, wobble));

        const gradAngle = -t * 0.4 + strandPhase;
        const gx1 = cx + Math.cos(gradAngle) * size * 0.4;
        const gy1 = cy + Math.sin(gradAngle) * size * 0.4;
        const gx2 = cx - Math.cos(gradAngle) * size * 0.4;
        const gy2 = cy - Math.sin(gradAngle) * size * 0.4;

        const grad = ctx.createLinearGradient(gx1, gy1, gx2, gy2);
        const bAlpha = (0.2 + p * 0.35) * (0.5 + Math.sin(t * 2 + s) * 0.3);
        grad.addColorStop(0, `rgba(217, 70, 239, ${bAlpha})`);
        grad.addColorStop(0.5, `rgba(139, 92, 246, ${bAlpha * 0.8})`);
        grad.addColorStop(1, `rgba(6, 182, 212, ${bAlpha})`);

        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5 + p * 1;
        ctx.stroke(path);
      }

      // Center glow
      const glowR = baseR * 0.5 * (0.8 + p * 0.3);
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR);
      const glowAlpha = isDark ? 0.06 + p * 0.08 : 0.03 + p * 0.04;
      glow.addColorStop(0, `rgba(217, 70, 239, ${glowAlpha})`);
      glow.addColorStop(0.5, `rgba(139, 92, 246, ${glowAlpha * 0.5})`);
      glow.addColorStop(1, "transparent");
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, size, size);

      // Outer pulse ring (subtle)
      const pulseR = baseR + 20 + Math.sin(t * 1.5) * 8 + p * 15;
      const pulseAlpha = (0.04 + p * 0.06) * (0.5 + Math.sin(t * 2) * 0.3);
      ctx.beginPath();
      ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(139, 92, 246, ${pulseAlpha})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();

      animRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [size, progress, isDark]);

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size }}
      />
    </div>
  );
}
