"""Render the RSASE manuscript (vault Markdown) to a single-column submission PDF with xelatex.

Usage: python pilot/paper_pdf.py --md <paper-rsase-draft.md> --figs <figdir> --out <paper-rsase-draft.pdf>
Needs only the Python standard library and /Library/TeX/texbin/xelatex (no pandoc).
Layout: 12 pt, double spacing, continuous line numbers, page numbers, figures and tables embedded
after the paragraph that first cites them. [OWNER: ...] markers are printed in red.
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

XELATEX = "/Library/TeX/texbin/xelatex"

PREAMBLE = r"""\documentclass[12pt]{article}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{fontspec}
\usepackage{amsmath}
\usepackage{unicode-math}
\setmainfont{Times New Roman}
\setmathfont{STIX Two Math}
\setmonofont{Menlo}[Scale=0.85]
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{float}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false}
\usepackage{xcolor}
\usepackage{setspace}
\usepackage[running]{lineno}
\usepackage{ragged2e}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\urlstyle{same}
\def\UrlBreaks{\do\/\do-\do.\do_\do=\do?\do&\do:\do;\do(\do)\do[\do]}
\newcommand{\owner}[1]{\textcolor{red}{\textbf{[#1]}}}
\newcolumntype{Y}{>{\RaggedRight\arraybackslash}X}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\pagestyle{plain}
\begin{document}
\doublespacing
\linenumbers
"""


def esc(t):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in t)


def inline(t):
    """Markdown inline -> LaTeX. Protect special spans with placeholders, escape the rest."""
    keep = []

    def ph(s):
        keep.append(s)
        return f"\x00{len(keep) - 1}\x00"

    t = re.sub(r"\[OWNER:([^\]]*)\]", lambda m: ph(r"\owner{OWNER:" + inline(m.group(1)) + "}"), t)
    t = re.sub(r"\[OWNER\]", lambda m: ph(r"\owner{OWNER}"), t)
    t = re.sub(r"`([^`]+)`", lambda m: ph(r"\texttt{" + esc(m.group(1)) + "}"), t)
    t = re.sub(r"(https?://[^\s)]+?)(?=[.,;]?(?:\s|$|\)))", lambda m: ph(r"\url{" + m.group(1).replace("%", r"\%").replace("#", r"\#") + "}"), t)
    t = re.sub(r"\*p\*_det", lambda m: ph(r"$p_{\mathrm{det}}$"), t)
    t = re.sub(r"\*\*([^*]+)\*\*", lambda m: ph(r"\textbf{" + inline(m.group(1)) + "}"), t)
    t = re.sub(r"\*([^*\s][^*]*)\*", lambda m: ph(r"\emph{" + inline(m.group(1)) + "}"), t)
    t = re.sub(r"\^([a-z*,]+)", lambda m: ph(r"\textsuperscript{" + esc(m.group(1)) + "}"), t)
    t = t.replace("\\*", "*")
    t = esc(t)
    t = t.replace("ᵀ", r"\textsuperscript{T}").replace("≥", r"\ensuremath{\geq}").replace("≤", r"\ensuremath{\leq}")
    for _ in range(3):  # nested placeholders
        t = re.sub("\x00(\\d+)\x00", lambda m: keep[int(m.group(1))], t)
    return t


def table(rows, caption):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|[-|\s:]+\|$", r.strip())]
    head, body = cells[0], cells[1:]
    n = len(head)
    numeric = all(re.match(r"^[−\-+0-9.() ]+$|^B\d$|^A$", r[0]) for r in body) and n > 4
    if numeric:
        spec = "l" + "r" * (n - 1)
        env_open, env_close = r"\resizebox{\linewidth}{!}{\begin{tabular}{" + spec + "}", r"\end{tabular}}"
        size = r"\footnotesize"
    else:
        spec = "l" + "Y" * (n - 1) if n > 2 else "lY"
        env_open, env_close = r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\end{tabularx}"
        size = r"\small"
    out = [r"\begin{table}[H]", r"\begin{singlespace}", size, r"\caption*{" + caption + "}", env_open, r"\toprule",
           " & ".join(r"\textbf{" + inline(h) + "}" for h in head) + r" \\", r"\midrule"]
    out += [" & ".join(inline(c) for c in r) + r" \\" for r in body]
    out += [r"\bottomrule", env_close, r"\end{singlespace}", r"\end{table}"]
    return "\n".join(out)


def figure(path, num, caption):
    return "\n".join([r"\begin{figure}[H]", r"\centering", rf"\includegraphics[width=\linewidth,height=0.42\textheight,keepaspectratio]{{{path}}}",
                      r"\begin{singlespace}", r"\caption*{" + caption + "}", r"\end{singlespace}", r"\end{figure}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    md = Path(a.md).read_text()
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.S)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    body, caps = md.split("## Figure captions")
    capd = {}
    for m in re.finditer(r"\*\*Fig\. (\d+)\. ([^*]+)\*\*\s*(.*)", caps):
        capd[int(m.group(1))] = r"\textbf{Fig. " + m.group(1) + ". " + inline(m.group(2).strip()) + "} " + inline(m.group(3).strip())
    figdir = Path(a.figs)
    placed = set()
    out = [PREAMBLE]
    lines = body.strip().split("\n")
    i = 0
    pending_caption = None
    para = []
    in_refs = False
    in_title = True

    def flush():
        nonlocal para
        if not para:
            return
        text = " ".join(para)
        para = []
        if in_title:
            out.append(r"\begin{center}" + inline(text) + r"\end{center}" + "\n")
        elif in_refs:
            out.append(r"\noindent\hangindent=1.5em\hangafter=1 " + inline(text) + "\n")
        else:
            out.append(inline(text) + "\n")
        for m in re.finditer(r"Figs?\. (\d+)(?: and (\d+))?", text):
            for g in m.groups():
                if g and int(g) not in placed and int(g) in capd:
                    placed.add(int(g))
                    ext = "pdf" if (figdir / f"Figure_{g}.pdf").exists() else "png"
                    out.append(figure(str(figdir / f"Figure_{g}.{ext}"), int(g), capd[int(g)]))

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("# ") and not s.startswith("## "):
            flush()
            out.append(r"\begin{center}{\LARGE\bfseries\begin{spacing}{1.2}" + inline(s[2:]) + r"\end{spacing}}\end{center}")
        elif s.startswith("## Graphical abstract"):
            flush()
            out.append(r"\section*{Graphical abstract}")
            out.append(r"\begin{center}\includegraphics[width=\linewidth]{" + str(figdir / "Graphical_abstract.pdf") + r"}\end{center}")
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        elif s.startswith("### "):
            flush()
            if s[4:].strip() == "Web references":
                out.append(r"\subsection*{Web references}")
            else:
                out.append(r"\subsection*{" + inline(s[4:]) + "}")
        elif s.startswith("## "):
            flush()
            title = s[3:]
            in_title = False
            in_refs = title.strip() == "References"
            if title.strip() == "Highlights":
                out.append(r"\newpage")
            out.append(r"\section*{" + inline(title) + "}")
        elif s.startswith("**Table ") and i + 2 < len(lines):
            flush()
            m = re.match(r"\*\*(Table \d+\.)\*\*\s*(.*)", s)
            pending_caption = r"\textbf{" + m.group(1) + "} " + inline(m.group(2))
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(table(rows, pending_caption or ""))
            pending_caption = None
            continue
        elif s.startswith("- "):
            flush()
            out.append(r"\begin{itemize}")
            while i < len(lines) and lines[i].strip().startswith("- "):
                out.append(r"\item " + inline(lines[i].strip()[2:]))
                i += 1
            out.append(r"\end{itemize}")
            continue
        elif re.match(r"^\*p\*_det = σ", s):
            flush()
            out.append(r"\begin{equation}p_{\mathrm{det}} = \sigma(\symbf{w}^{\top}\symbf{q}),\qquad \varepsilon = \sigma(\symbf{v}^{\top}\symbf{q})\end{equation}")
        elif s == "":
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    missing = sorted(set(capd) - placed)
    for g in missing:  # figures never cited in text: append at end
        out.append(figure(str(figdir / f"Figure_{g}.pdf"), g, capd[g]))
    out.append(r"\end{document}")
    tex = "\n".join(out)
    tex = tex.replace(r"\textbf{Authors:} ", r"\large ", 1)
    with tempfile.TemporaryDirectory() as td:
        tf = Path(td) / "paper.tex"
        tf.write_text(tex)
        for _ in range(2):
            r = subprocess.run([XELATEX, "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=td, capture_output=True, text=True)
            if r.returncode != 0:
                Path(a.out).with_suffix(".log.txt").write_text(r.stdout[-6000:])
                raise SystemExit(f"xelatex failed; see {Path(a.out).with_suffix('.log.txt')}")
        shutil.copy(Path(td) / "paper.pdf", a.out)
        log = (Path(td) / "paper.log").read_text(errors="ignore")
        miss = sorted(set(re.findall(r"Missing character: There is no (.) ", log)))
        print("pdf:", a.out, "| figures placed:", sorted(placed), "| appended:", missing, "| missing glyphs:", miss)


if __name__ == "__main__":
    main()
