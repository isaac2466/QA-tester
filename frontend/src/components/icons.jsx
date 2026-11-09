// Íconos en línea (trazo 2px, estilo "lucide"). Heredan el color del texto.
const Svg = ({ children, size = 18, ...p }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...p}>
    {children}
  </svg>
);

export const Camera = (p) => (
  <Svg {...p}><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z" /><circle cx="12" cy="13" r="3" /></Svg>
);
export const Compare = (p) => (
  <Svg {...p}><rect x="2" y="4" width="13" height="10" rx="2" /><path d="M9 18h11a2 2 0 0 0 2-2V9" /></Svg>
);
export const History = (p) => (
  <Svg {...p}><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l3 2" /></Svg>
);
export const Upload = (p) => (
  <Svg {...p}><path d="M12 15V3" /><path d="m7 8 5-5 5 5" /><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /></Svg>
);
export const Play = (p) => <Svg {...p}><path d="m7 4 13 8-13 8z" /></Svg>;
export const Eye = (p) => (
  <Svg {...p}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></Svg>
);
export const Columns = (p) => <Svg {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M12 4v16" /></Svg>;
export const Layers = (p) => (
  <Svg {...p}><path d="m12 3 9 5-9 5-9-5z" /><path d="m3 13 9 5 9-5" /></Svg>
);
export const Diff = (p) => (
  <Svg {...p}><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /></Svg>
);
export const Zoom = (p) => (
  <Svg {...p}><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3M11 8v6M8 11h6" /></Svg>
);
export const Flag = (p) => <Svg {...p}><path d="M4 22V4s1-1 4-1 5 2 8 2 4-1 4-1v11s-1 1-4 1-5-2-8-2-4 1-4 1" /></Svg>;
export const Target = (p) => (
  <Svg {...p}><circle cx="12" cy="12" r="7" /><circle cx="12" cy="12" r="2" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3" /></Svg>
);
export const Download = (p) => (
  <Svg {...p}><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /></Svg>
);
export const Info = (p) => <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M12 16v-4M12 8h.01" /></Svg>;
export const Chevron = (p) => <Svg {...p}><path d="m9 6 6 6-6 6" /></Svg>;
export const Close = (p) => <Svg {...p}><path d="M18 6 6 18M6 6l12 12" /></Svg>;
export const EyeOff = (p) => (
  <Svg {...p}><path d="M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-2.2 3.1M6.6 6.6C3.9 8.3 2 12 2 12s3.5 7 10 7a9.7 9.7 0 0 0 5.4-1.6" /><path d="m2 2 20 20" /><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" /></Svg>
);
export const Sparkles = (p) => (
  <Svg {...p}><path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z" /><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z" /></Svg>
);
export const Wand = (p) => (
  <Svg {...p}><path d="m15 4 5 5L8 21H3v-5z" /><path d="M12 7l5 5" /><path d="M19 2v3M17.5 3.5h3" /></Svg>
);
export const Check = (p) => <Svg {...p}><path d="m5 12 5 5L20 7" /></Svg>;
export const Logo = ({ size = 30 }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
    <path d="M10 6 18 14 10 22 2 14z" fill="#7c3aed" />
    <path d="M22 6 30 14 22 22 14 14z" fill="#a78bfa" opacity=".9" />
  </svg>
);
