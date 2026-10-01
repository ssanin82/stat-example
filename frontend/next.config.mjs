// Bind the dev server to localhost only via the npm script
// (`next dev -p 8001` -- by default Next.js binds 127.0.0.1).
// API keys live in this process; do NOT change to 0.0.0.0.

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Disable telemetry phone-home; this is a local-only operator tool.
  experimental: {
    // (intentionally empty for now)
  },
};

export default nextConfig;
