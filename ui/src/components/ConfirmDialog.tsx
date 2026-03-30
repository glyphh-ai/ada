import { useCallback, useEffect, useRef } from "react";
import { HiOutlineExclamation } from "react-icons/hi";

type Props = {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "danger" | "warning";
  onConfirm: () => void;
  onCancel: () => void;
};

export default function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "danger",
  onConfirm,
  onCancel,
}: Props) {
  const cancelRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) cancelRef.current?.focus();
  }, [open]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    },
    [onCancel],
  );

  if (!open) return null;

  const isDanger = variant === "danger";

  return (
    <div
      className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
      onClick={onCancel}
      onKeyDown={onKeyDown}
    >
      <div
        className="w-full max-w-sm bg-[var(--bg)] text-[var(--fg)] border border-[var(--border)] rounded-2xl shadow-2xl p-6 space-y-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3">
          <div
            className={`flex items-center justify-center w-10 h-10 rounded-full ${
              isDanger
                ? "bg-red-500/10 text-red-400"
                : "bg-amber-500/10 text-amber-400"
            }`}
          >
            <HiOutlineExclamation size={22} />
          </div>
          <div className="text-base font-semibold">{title}</div>
        </div>

        <p className="text-sm text-[var(--muted)] leading-relaxed">{message}</p>

        <div className="flex items-center justify-end gap-2 pt-2">
          <button
            ref={cancelRef}
            onClick={onCancel}
            className="px-4 py-2 text-xs rounded-xl border border-[var(--border)] text-[var(--muted)] hover:text-[var(--fg)] hover:border-[var(--fg)]/30 transition cursor-pointer"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className={`px-4 py-2 text-xs font-semibold rounded-xl shadow transition cursor-pointer ${
              isDanger
                ? "bg-red-500 text-white hover:bg-red-600 shadow-red-500/30"
                : "bg-amber-500 text-white hover:bg-amber-600 shadow-amber-500/30"
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
