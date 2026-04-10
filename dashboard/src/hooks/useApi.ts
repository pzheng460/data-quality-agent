const API = import.meta.env.VITE_API_URL || 'http://localhost:8001'

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init)
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`)
  return res.json()
}

export function apiUrl(path: string): string {
  return `${API}${path}`
}
