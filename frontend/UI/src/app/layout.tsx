// frontend/UI/src/app/layout.tsx
import type { Metadata } from "next";
import { IBM_Plex_Mono } from "next/font/google";
import "../styles/globals.css";

const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "700"] });

export const metadata: Metadata = {
  title: "GeneSpectra | AI Analysis Platform",
  description: "Advanced deep learning interpretation of DNA sequence pathogenicity.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={plexMono.className}>{children}</body>
    </html>
  );
}