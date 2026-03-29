/**
 * RESONATE — Skeleton Loading Row.
 * Shimmer gradient animation for a sleek loading state.
 */

import { useTheme } from "../theme/ThemeProvider";

function ShimmerBlock({ width, height = 10, delay = 0, isDark }) {
  return (
    <div style={{
      width, height, borderRadius: 4,
      background: isDark
        ? "linear-gradient(90deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.04) 100%)"
        : "linear-gradient(90deg, rgba(0,0,0,0.04) 0%, rgba(0,0,0,0.07) 50%, rgba(0,0,0,0.04) 100%)",
      backgroundSize: "200% 100%",
      animation: `shimmer 1.8s ease-in-out ${delay}s infinite`,
    }} />
  );
}

export function SkeletonRow({ index = 0 }) {
  const { mode } = useTheme();
  const isDark = mode === "dark";
  const delay = index * 0.06;

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "22px 30px 1fr 54px 40px 40px 42px",
      gap: 6,
      padding: "11px 14px",
      borderBottom: isDark ? "1px solid rgba(255,255,255,0.03)" : "1px solid rgba(0,0,0,0.03)",
      animation: `staggerFadeUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) ${delay}s both`,
    }}>
      <ShimmerBlock width={13} height={13} delay={delay} isDark={isDark} />
      <ShimmerBlock width={11} height={11} delay={delay + 0.02} isDark={isDark} />
      <div>
        <ShimmerBlock width="70%" height={11} delay={delay + 0.04} isDark={isDark} />
        <div style={{ marginTop: 4 }}>
          <ShimmerBlock width="35%" height={8} delay={delay + 0.06} isDark={isDark} />
        </div>
      </div>
      <ShimmerBlock width={34} height={10} delay={delay + 0.08} isDark={isDark} />
      <ShimmerBlock width={22} height={10} delay={delay + 0.1} isDark={isDark} />
      <ShimmerBlock width={22} height={10} delay={delay + 0.12} isDark={isDark} />
      <ShimmerBlock width={26} height={10} delay={delay + 0.14} isDark={isDark} />
    </div>
  );
}
