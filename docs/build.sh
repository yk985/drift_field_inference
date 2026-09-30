#!/bin/sh
# Rebuild the PDFs in this folder (needs pdflatex; MiKTeX or TeX Live).
# Figures come from ../outputs -- run the scripts named in each document first
# if you want them regenerated.
cd "$(dirname "$0")"
for doc in dfi_framework_reference binned_km_error_analysis drift_field_inference_report; do
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null   # second pass: TOC, refs
  echo "-> docs/$doc.pdf"
done
rm -f *.aux *.out *.toc *.log
