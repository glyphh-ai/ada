import React from "react";

type Props = {
  title: string | React.ReactNode;
  subtitle?: string;
  actions?: React.ReactNode;
};

export default function PageHeader({ title, subtitle, actions }: Props) {
  return (
    <div className="pt-6 bg-[var(--page-header)] border-b border-[var(--page-header-border)]">
      <div className="flex items-center justify-between gap-3 pb-2">
        <div className="px-4">
          <h1 className="text-lg font-semibold">{title}</h1>
          {subtitle && <p className="text-sm text-[var(--muted)]">{subtitle}</p>}
        </div>
        {actions && <div className="flex items-center gap-2 pr-4">{actions}</div>}
      </div>
    </div>
  );
}
