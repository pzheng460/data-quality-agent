"""Arxiv data cleaning pipeline — 3-stage architecture.

Stage 1 (Ingestion):   Download raw LaTeX from arxiv e-print
Stage 2 (Extraction):  Convert LaTeX → clean markdown via pandoc
Stage 3 (Curation):    Quality filtering, dedup, contamination via dq
"""
