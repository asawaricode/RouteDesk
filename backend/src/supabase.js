import { createClient } from "@supabase/supabase-js";

/**
 * Normalizes the Supabase URL by trimming whitespace and removing any
 * trailing '/rest/v1' or trailing slashes that users might copy from the Supabase API dashboard.
 *
 * @param {string} url
 * @returns {string}
 */
export function normalizeSupabaseUrl(url) {
  if (!url || typeof url !== "string") return "";
  return url.trim().replace(/\/rest\/v1\/?$/, "").replace(/\/+$/, "");
}

/**
 * Validates and retrieves the Supabase configuration from environment variables.
 * Fails with a clear, descriptive error if either SUPABASE_URL or SUPABASE_SECRET_KEY is missing.
 *
 * @returns {{ supabaseUrl: string, supabaseSecretKey: string }}
 */
export function getSupabaseConfig() {
  const rawUrl = process.env.SUPABASE_URL;
  const secretKey = process.env.SUPABASE_SECRET_KEY;

  if (!rawUrl || !rawUrl.trim()) {
    throw new Error(
      "[Supabase Config Error] Missing required environment variable 'SUPABASE_URL'. Please set SUPABASE_URL in backend/.env"
    );
  }

  if (!secretKey || !secretKey.trim()) {
    throw new Error(
      "[Supabase Config Error] Missing required environment variable 'SUPABASE_SECRET_KEY'. Please set SUPABASE_SECRET_KEY in backend/.env"
    );
  }

  const supabaseUrl = normalizeSupabaseUrl(rawUrl);

  return { supabaseUrl, supabaseSecretKey: secretKey.trim() };
}

// Read configuration securely on module initialization
const { supabaseUrl, supabaseSecretKey } = getSupabaseConfig();

/**
 * Supabase client instance with administrative access (service_role secret key).
 * Bypasses Row Level Security (RLS) — must only be used within the Express backend.
 */
export const supabase = createClient(supabaseUrl, supabaseSecretKey, {
  auth: {
    persistSession: false,
    autoRefreshToken: false,
  },
});

export default supabase;
