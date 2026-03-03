/**
 * DocLD API provider for table extraction.
 * Uses sync POST /api/parse/image-tables: single request, no queue or polling.
 */

import fs from 'node:fs';
import path from 'node:path';

const DOCLD_API_KEY = process.env.DOCLD_API_KEY;
const DOCLD_BASE_URL = process.env.DOCLD_BASE_URL || 'https://docld.com';

export interface DocLDConfig {
  apiKey?: string;
  baseUrl?: string;
}

function getConfig(config?: DocLDConfig): { apiKey: string; baseUrl: string } {
  const apiKey = config?.apiKey || DOCLD_API_KEY;
  const baseUrl = (config?.baseUrl || DOCLD_BASE_URL).replace(/\/$/, '');
  if (!apiKey) {
    throw new Error(
      'DOCLD_API_KEY is required. Set it in .env or pass apiKey to extractTableFromImage.'
    );
  }
  return { apiKey, baseUrl };
}

/**
 * Extract table HTML from an image using DocLD sync image-tables API.
 * Single request: no upload, no polling.
 */
export async function extractTableFromImage(
  imagePath: string,
  config?: DocLDConfig
): Promise<string> {
  const { apiKey, baseUrl } = getConfig(config);

  const fullPath = path.resolve(imagePath);
  if (!fs.existsSync(fullPath)) {
    throw new Error(`Image not found: ${fullPath}`);
  }

  const buffer = fs.readFileSync(fullPath);
  const filename = path.basename(fullPath) || 'table.png';

  const formData = new FormData();
  formData.append('file', new Blob([buffer]), filename);
  formData.append(
    'config',
    JSON.stringify({ tableOutputMode: 'html' })
  );

  const res = await fetch(`${baseUrl}/api/parse/image-tables`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      `Image-tables API failed: ${res.status} ${(err as any)?.error?.message || res.statusText}`
    );
  }

  const json = await res.json();
  const payload = json?.data ?? json;
  const tables = payload?.tables;

  if (Array.isArray(tables) && tables.length > 0) {
    const first = tables[0];
    const content = first?.content;
    if (typeof content === 'string') return content;
  }

  return '';
}
