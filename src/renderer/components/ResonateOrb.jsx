/**
 * RESONATE — Animated Orb.
 * Organic swirling aurora-like shape with particles.
 * Intensifies as analysis progress increases.
 * Brand gradient: pink #D946EF → purple #8B5CF6 → cyan #06B6D4
 */

import { useEffect, useRef } from "react";
import { useTheme } from "../theme/ThemeProvider";

// Smooth noise function for organic motion
function noise(x, y, t) {
  return (
    Math.sin(x * 1.2 + t * 0.7) * Math.cos(y * 0.9 + t * 0.5) * 0.5 +
    Math.sin(x * 2.5 + y * 1.8 + t * 1.1) * 0.3 +
    Math.cos(x * 0.6 - t * 0.3) * Math.sin(y * 1.5 + t * 0.8) * 0.2
  );
}

// Generate fluid blob path with organic deformation
function blobPath(cx, cy, baseR, points, phase, wobble, t) {
  const pts = [];
  for (let i = 0; i <= points; i++) {
    const angle = (i / points) * Math.PI * 2;
    const n = noise(Math.cos(angle) * 2, Math.sin(angle) * 2, phase);
    const r = baseR +
      Math.sin(angle * 3 + phase) * wobble * 0.5 +
      Math.sin(angle * 5 + phase * 1.3) * wobble * 0.25 +
      Math.cos(angle * 2 + phase * 0.8) * wobble * 0.35 +
      n * wobble * 0.4;
    pts.push([cx + Math.cos(angle) * r, cy + Math.sin(angle) * r]);
  }
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2;
    const my = (pts[i][1] + pts[i + 1][1]) / 2;
    d += ` Q ${pts[i][0]} ${pts[i][1]} ${mx} ${my}`;
  }
  d += " Z";
  return d;
}

// Particle system for sparkle effect
class Particle {
  constructor(cx, cy, baseR) {
    this.reset(cx, cy, baseR);
  }
  reset(cx, cy, baseR) {
    const angle = Math.random() * Math.PI * 2;
    const dist = baseR * (0.6 + Math.random() * 0.5);
    this.x = cx + Math.cos(angle) * dist;
    this.y = cy + Math.sin(angle) * dist;
    this.vx = (Math.random() - 0.5) * 0.3;
    this.vy = (Math.random() - 0.5) * 0.3;
    this.life = 1;
    this.decay = 0.003 + Math.random() * 0.008;
    this.size = 0.5 + Math.random() * 1.5;
    // 0 = pink, 0.5 = purple, 1 = cyan
    this.color = Math.random();
  }
  update(cx, cy, baseR) {
    this.x += this.vx;
    this.y += this.vy;
    this.life -= this.decay;
    if (this.life <= 0) this.reset(cx, cy, baseR);
  }
}

export function ResonateOrb({ progress = 0, size = 320 }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const particlesRef = useRef(null);
  const progressRef = useRef(progress);
  const isDarkRef = useRef(false);
  const { mode } = useTheme();
  const isDark = mode === "dark";

  // Keep refs in sync without re-running effect
  progressRef.current = progress;
  isDarkRef.current = isDark;

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
    const startTime = Date.now();

    // Initialize particles
    if (!particlesRef.current) {
      particlesRef.current = Array.from({ length: 40 }, () => new Particle(cx, cy, baseR));
    }

    function draw() {
      const t = (Date.now() - startTime) / 1000;
      const p = progressRef.current / 100;
      ctx.clearRect(0, 0, size, size);

      // ── 1. Deep background glow ──────────────────────────────────────
      const bgGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, baseR * 1.8);
      const dk = isDarkRef.current;
      const bgA = dk ? 0.04 + p * 0.06 : 0.02 + p * 0.03;
      bgGlow.addColorStop(0, `rgba(139, 92, 246, ${bgA})`);
      bgGlow.addColorStop(0.4, `rgba(217, 70, 239, ${bgA * 0.6})`);
      bgGlow.addColorStop(0.7, `rgba(6, 182, 212, ${bgA * 0.3})`);
      bgGlow.addColorStop(1, "transparent");
      ctx.fillStyle = bgGlow;
      ctx.fillRect(0, 0, size, size);

      // ── 2. Aurora flowing strands (outer) ────────────────────────────
      const outerCount = 16;
      for (let s = 0; s < outerCount; s++) {
        const strandPhase = (s / outerCount) * Math.PI * 2;
        const speed = 0.3 + s * 0.06 + p * 0.2;
        const phase = t * speed + strandPhase;
        const wobble = 14 + p * 25 + Math.sin(t * 0.4 + s * 0.7) * 8;
        const r = baseR - 10 + s * 2.2;

        const path = new Path2D(blobPath(cx, cy, r, 72, phase, wobble, t));

        // Rotating gradient direction
        const gradAngle = t * 0.25 + strandPhase * 0.5;
        const gx1 = cx + Math.cos(gradAngle) * size * 0.45;
        const gy1 = cy + Math.sin(gradAngle) * size * 0.45;
        const gx2 = cx - Math.cos(gradAngle) * size * 0.45;
        const gy2 = cy - Math.sin(gradAngle) * size * 0.45;

        const grad = ctx.createLinearGradient(gx1, gy1, gx2, gy2);
        const alpha = (0.04 + (s / outerCount) * 0.1) * (0.5 + p * 0.5);
        const breathe = 0.7 + Math.sin(t * 1.5 + s * 0.4) * 0.3;
        const a = alpha * breathe;
        grad.addColorStop(0, `rgba(217, 70, 239, ${a})`);
        grad.addColorStop(0.35, `rgba(139, 92, 246, ${a * 0.85})`);
        grad.addColorStop(0.65, `rgba(6, 182, 212, ${a * 0.9})`);
        grad.addColorStop(1, `rgba(217, 70, 239, ${a * 0.7})`);

        ctx.strokeStyle = grad;
        ctx.lineWidth = 1 + p * 0.6;
        ctx.stroke(path);
      }

      // ── 3. Bright inner energy strands ───────────────────────────────
      const innerCount = 8;
      for (let s = 0; s < innerCount; s++) {
        const strandPhase = (s / innerCount) * Math.PI * 2;
        const phase = t * (0.5 + s * 0.1 + p * 0.3) + strandPhase + Math.PI * 0.5;
        const wobble = 10 + p * 30 + Math.sin(t * 0.6 + s * 1.1) * 10;
        const r = baseR - 6 + s * 2.8;

        const path = new Path2D(blobPath(cx, cy, r, 72, phase, wobble, t));

        const gradAngle = -t * 0.35 + strandPhase;
        const gx1 = cx + Math.cos(gradAngle) * size * 0.4;
        const gy1 = cy + Math.sin(gradAngle) * size * 0.4;
        const gx2 = cx - Math.cos(gradAngle) * size * 0.4;
        const gy2 = cy - Math.sin(gradAngle) * size * 0.4;

        const grad = ctx.createLinearGradient(gx1, gy1, gx2, gy2);
        const pulse = 0.5 + Math.sin(t * 2.5 + s * 0.8) * 0.4;
        const a = (0.15 + p * 0.4) * pulse;
        grad.addColorStop(0, `rgba(217, 70, 239, ${a})`);
        grad.addColorStop(0.5, `rgba(139, 92, 246, ${a * 0.75})`);
        grad.addColorStop(1, `rgba(6, 182, 212, ${a})`);

        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5 + p * 1.2;
        ctx.stroke(path);
      }

      // ── 4. Energetic core glow (pulsating) ──────────────────────────
      const coreSize = baseR * (0.35 + p * 0.15) * (0.9 + Math.sin(t * 2) * 0.1);
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreSize);
      const coreA = dk ? 0.08 + p * 0.12 : 0.04 + p * 0.06;
      core.addColorStop(0, `rgba(255, 255, 255, ${coreA * 0.3})`);
      core.addColorStop(0.2, `rgba(217, 70, 239, ${coreA})`);
      core.addColorStop(0.5, `rgba(139, 92, 246, ${coreA * 0.6})`);
      core.addColorStop(1, "transparent");
      ctx.fillStyle = core;
      ctx.fillRect(0, 0, size, size);

      // ── 5. Particles ────────────────────────────────────────────────
      const particles = particlesRef.current;
      const particleCount = Math.floor(15 + p * 25);
      for (let i = 0; i < Math.min(particleCount, particles.length); i++) {
        const pt = particles[i];
        pt.update(cx, cy, baseR);

        const r = pt.color < 0.33 ? 217 : pt.color < 0.66 ? 139 : 6;
        const g = pt.color < 0.33 ? 70 : pt.color < 0.66 ? 92 : 182;
        const b = pt.color < 0.33 ? 239 : pt.color < 0.66 ? 246 : 212;
        const pa = pt.life * (0.3 + p * 0.5) * (dk ? 1 : 0.6);

        ctx.beginPath();
        ctx.arc(pt.x, pt.y, pt.size * (0.5 + p * 0.5), 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${pa})`;
        ctx.fill();

        // Particle glow
        if (pa > 0.2) {
          const ptGlow = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, pt.size * 3);
          ptGlow.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${pa * 0.15})`);
          ptGlow.addColorStop(1, "transparent");
          ctx.fillStyle = ptGlow;
          ctx.fillRect(pt.x - pt.size * 3, pt.y - pt.size * 3, pt.size * 6, pt.size * 6);
        }
      }

      // ── 6. Outer pulse rings ────────────────────────────────────────
      for (let r = 0; r < 3; r++) {
        const ringPhase = t * (1.2 + r * 0.3) + r * Math.PI * 0.67;
        const ringR = baseR + 18 + r * 8 + Math.sin(ringPhase) * 10 + p * 20;
        const ringA = (0.03 + p * 0.05) * (0.4 + Math.sin(ringPhase + Math.PI / 2) * 0.4);
        const colors = [
          `rgba(217, 70, 239, ${ringA})`,
          `rgba(139, 92, 246, ${ringA})`,
          `rgba(6, 182, 212, ${ringA})`,
        ];
        ctx.beginPath();
        ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
        ctx.strokeStyle = colors[r];
        ctx.lineWidth = 0.6 + p * 0.4;
        ctx.stroke();
      }

      // ── 7. Energy arcs (progress-responsive) ───────────────────────
      if (p > 0.1) {
        const arcCount = Math.floor(3 + p * 4);
        for (let a = 0; a < arcCount; a++) {
          const arcPhase = t * (0.8 + a * 0.15) + a * Math.PI * 2 / arcCount;
          const arcR = baseR * (0.7 + Math.sin(arcPhase * 0.5) * 0.2);
          const arcStart = arcPhase;
          const arcLen = Math.PI * (0.3 + p * 0.4 + Math.sin(t * 1.5 + a) * 0.15);
          const arcA = (0.1 + p * 0.25) * (0.5 + Math.sin(t * 3 + a * 1.3) * 0.4);

          ctx.beginPath();
          ctx.arc(cx, cy, arcR, arcStart, arcStart + arcLen);
          const arcGrad = ctx.createLinearGradient(
            cx + Math.cos(arcStart) * arcR, cy + Math.sin(arcStart) * arcR,
            cx + Math.cos(arcStart + arcLen) * arcR, cy + Math.sin(arcStart + arcLen) * arcR
          );
          arcGrad.addColorStop(0, `rgba(217, 70, 239, 0)`);
          arcGrad.addColorStop(0.3, `rgba(217, 70, 239, ${arcA})`);
          arcGrad.addColorStop(0.7, `rgba(6, 182, 212, ${arcA})`);
          arcGrad.addColorStop(1, `rgba(6, 182, 212, 0)`);
          ctx.strokeStyle = arcGrad;
          ctx.lineWidth = 1.5 + p * 1;
          ctx.lineCap = "round";
          ctx.stroke();
        }
      }

      animRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [size]);

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size }}
      />
    </div>
  );
}
