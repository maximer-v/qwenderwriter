# Paper

`Qwenderwriter.pdf` is the paper describing the system, its guarantees and the evaluation reported in
[`docs/EVALUATION.md`](../docs/EVALUATION.md).

`source/` holds the LaTeX source: `main.tex`, the generated `numbers.tex` (every number quoted in the paper, pulled
from the build artefacts), `references.bib` with its compiled `main.bbl`, and the figures. Build it with any TeX
distribution, for example:

```bash
cd source
tectonic main.tex        # or: pdflatex main && pdflatex main
```

The paper is released under CC BY-NC 4.0, like the rest of the repository.
