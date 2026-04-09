import { useState, useEffect } from 'react'

const cache = new Map<string, unknown>()

export function useData<T>(path: string): {
  data: T | null
  loading: boolean
  error: Error | null
} {
  const [data, setData] = useState<T | null>(() =>
    cache.has(path) ? (cache.get(path) as T) : null
  )
  const [loading, setLoading] = useState(!cache.has(path))
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (cache.has(path)) {
      setData(cache.get(path) as T)
      setLoading(false)
      return
    }

    setLoading(true)
    fetch(`./data/${path}`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`)
        return res.json()
      })
      .then((json) => {
        cache.set(path, json)
        setData(json)
        setLoading(false)
      })
      .catch((err) => {
        setError(err)
        setLoading(false)
      })
  }, [path])

  return { data, loading, error }
}

