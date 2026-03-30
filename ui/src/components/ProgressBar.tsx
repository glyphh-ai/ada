/**
 * Terminal-style progress bar matching glyphh-studio's monospace █░ aesthetic.
 *
 * Renders: [████████░░░░] 82%  label
 */

import { useState, useEffect } from "react";

type Props = {
  /** 0.0 – 1.0 */
  progress: number;
  /** e.g. "42 / 100 records" */
  label?: string;
  /** Total bar width in characters (default 16) */
  width?: number;
  /** Show indeterminate animation when progress is unknown */
  indeterminate?: boolean;
};

function bar(progress: number, width: number): string {
  const filled = Math.round(progress * width);
  return "\u2588".repeat(filled) + "\u2591".repeat(width - filled);
}

function indeterminateBar(tick: number, width: number): string {
  // Fill sweeps left-to-right, then resets and sweeps again
  const pos = tick % (width + 1);
  return "\u2588".repeat(pos) + "\u2591".repeat(width - pos);
}

export default function ProgressBar({
  progress,
  label,
  width = 16,
  indeterminate,
}: Props) {
  const pct = Math.round(Math.min(Math.max(progress, 0), 1) * 100);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!indeterminate) return;
    const id = setInterval(() => setTick((t) => t + 1), 150);
    return () => clearInterval(id);
  }, [indeterminate]);

  return (
    <div className="flex items-center gap-3 font-mono text-xs">
      <span className="text-[#A96BFE] tracking-tight">
        [{indeterminate ? indeterminateBar(tick, width) : bar(progress, width)}]
      </span>
      {!indeterminate && (
        <span className="text-[#A96BFE] font-semibold tabular-nums w-8 text-right">
          {pct}%
        </span>
      )}
      {label && (
        <span className="text-[var(--fg)] opacity-80 truncate">{label}</span>
      )}
      {(indeterminate || (progress > 0 && progress < 1)) && (
        <span className="inline-block w-2 h-2 rounded-full bg-[#A96BFE] animate-pulse" />
      )}
    </div>
  );
}
