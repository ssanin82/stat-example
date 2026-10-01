import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "DTC MM AS — Accounts",
  description: "Operator dashboard for the dtc-mm-as bot fleet.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      {/* suppressHydrationWarning: the Grammarly browser extension
          (and a few similar ones) inject ``data-new-gr-c-s-check-loaded``
          / ``data-gr-ext-installed`` attributes onto the body AFTER
          the server-rendered HTML lands. Without this prop, Next.js
          dev mode prints a noisy hydration-mismatch error -- the
          extension always wins the diff. The flag tells React to
          tolerate attribute mismatches on this specific element only.
          Recommended workaround per Next.js docs:
          https://nextjs.org/docs/messages/react-hydration-error */}
      <body className="min-h-screen theme-spaceship" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
