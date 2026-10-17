// Design tokens for the Agentic AI Learning Path Builder
// Based on premium technical aesthetic: Linear, Raycast, Notion inspiration

export const colors = {
  // Workspace and surfaces
  workspace: '#090a0c',       // Deep charcoal background
  surface1: '#111317',        // Primary container/card background
  surface2: '#181b20',        // Elevated surfaces
  surfaceActive: '#22262d',   // Hover/active state

  // Borders
  borderSubtle: '#1f242c',    // Subtle dividers
  borderStrong: '#2e3642',    // Focused borders

  // Typography
  textPrimary: '#f1f3f5',     // Headings and primary labels
  textSecondary: '#9aa2ad',   // Subtitles, descriptions
  textTertiary: '#5e6673',    // Disabled, placeholders, node IDs

  // Semantic state accents
  accentMastered: '#10b981',  // Emerald: Mastered concepts
  accentInProgress: '#0ea5e9', // Sky: Active learning
  accentAvailable: '#64748b',  // Slate: Unlocked and ready
  accentLocked: '#334155',    // Slate: Locked (missing prerequisites)

  // Gradients and effects
  gradientPrimary: 'from-slate-900 via-slate-800 to-slate-900',
  gradientOverlay: 'from-white/60 via-transparent to-white/60 dark:from-slate-900/60 dark:via-transparent dark:to-slate-900/60',
};

export const spacing = {
  xs: '0.25rem',
  sm: '0.5rem',
  md: '1rem',
  lg: '1.5rem',
  xl: '2rem',
  '2xl': '2.5rem',
  '3xl': '3rem',
  '4xl': '4rem',
};

export const radius = {
  sm: '0.25rem',   // 4px
  md: '0.375rem',  // 6px
  lg: '0.5rem',    // 8px
  xl: '0.75rem',   // 12px
};

export const shadows = {
  subtle: '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
  medium: '0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.15)',
  strong: '0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.15)',
};

export const typography = {
  // Font families would be set in CSS
  // Using Inter for body, Geist-inspired for headings
  fontWeight: {
    regular: 400,
    medium: 500,
    semiBold: 600,
    bold: 700,
    black: 900,
  },
  letterSpacing: {
    tight: '-0.015em',
    normal: '0em',
    wide: '0.025em',
  },
};

export const animations = {
  // Subtle, purposeful animations
  fast: '150ms ease-in-out',
  normal: '250ms ease-in-out',
  slow: '350ms cubic-bezier(0.4, 0, 0.2, 1)',
  spring: 'spring(stiffness: 100, damping: 20)',
};