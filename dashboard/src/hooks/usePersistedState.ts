import { useState, useEffect, useRef } from 'react'

/**
 * `useState` drop-in that reads/writes to localStorage under the given key.
 *
 * On first render: hydrates from localStorage if present, else uses `initial`.
 * On update: serializes the value (JSON) to localStorage so it survives reloads.
 *
 * Failures (disabled storage, parse errors) are ignored — it falls back to
 * in-memory state so components never crash due to storage issues.
 */
export function usePersistedState<T>(key: string, initial: T): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      return raw != null ? (JSON.parse(raw) as T) : initial
    } catch {
      return initial
    }
  })

  const firstWrite = useRef(true)
  useEffect(() => {
    try {
      // Skip the first effect run on initial mount — value already matches storage.
      // This prevents overwriting richer state from an alternate tab on open.
      if (firstWrite.current) {
        firstWrite.current = false
        return
      }
      localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // ignore — likely Safari private mode or quota exceeded
    }
  }, [key, value])

  return [value, setValue]
}
