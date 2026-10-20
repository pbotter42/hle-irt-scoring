"""Render the executed HLE notebook as offline HTML with native MathML."""
from pathlib import Path
import html
import json
import re
import markdown
from latex2mathml.converter import convert
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / "hle_scoring_walkthrough.ipynb").read_text())
fmt = HtmlFormatter()


def math_markdown(source):
    stash = {}

    def replace(match, display):
        token = "MATHPLACEHOLDER" + str(len(stash)) + "END"
        expression = match.group(1).strip()
        # latex2mathml supports array alignment; raw aligned emits literal ampersands.
        supported = expression.replace(r"\begin{aligned}", r"\begin{array}{rl}")
        supported = supported.replace(r"\end{aligned}", r"\end{array}")
        mathml = convert(supported, display="block" if display else "inline")
        # Native MathML carries structure; a label also preserves the source formula.
        mathml = mathml.replace("<math ", '<math aria-label="' + html.escape(expression, quote=True) + '" ', 1)
        stash[token] = ('<div class="equation">' + mathml + '</div>') if display else mathml
        return ("\n\n" + token + "\n\n") if display else token

    source = re.sub(r"\$\$(.*?)\$\$", lambda m: replace(m, True), source, flags=re.S)
    source = re.sub(r"\\\((.*?)\\\)", lambda m: replace(m, False), source, flags=re.S)
    rendered = markdown.markdown(source, extensions=["tables", "fenced_code"])
    for token, replacement in stash.items():
        rendered = rendered.replace("<p>" + token + "</p>", replacement).replace(token, replacement)
    return rendered


parts = []
figure = 0
for cell in nb["cells"]:
    source = "".join(cell["source"])
    if cell["cell_type"] == "markdown":
        parts.append(math_markdown(source))
        continue
    parts.append('<section class="cell"><div class="cell-label">Python cell ' +
                 str(cell["execution_count"]) + '</div>' +
                 highlight(source, PythonLexer(), fmt))
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            figure += 1
            alt = ("Illustrative IRT response curves showing difficulty and discrimination."
                   if figure == 1 else
                   "Three scatterplots compare HLE percent correct with full-data general scores.")
            parts.append('<img alt="' + alt + '" src="data:image/png;base64,' +
                         "".join(data["image/png"]) + '">')
        elif "text/html" in data:
            parts.append('<div class="output">' + "".join(data["text/html"]) + '</div>')
        elif "text/plain" in data:
            parts.append('<pre class="output">' + html.escape("".join(data["text/plain"])) + '</pre>')
        elif output.get("output_type") == "stream":
            parts.append('<pre class="output">' + html.escape("".join(output["text"])) + '</pre>')
        elif output.get("output_type") == "error":
            raise RuntimeError("Refusing to render a notebook with an execution error.")
    parts.append('</section>')
css = """
body{background:#fcfcfa;color:#24343b;font:18px/1.65 system-ui,sans-serif;margin:0}
main{max-width:1040px;margin:40px auto;padding:0 28px 64px}
h1{font-size:2.15rem;line-height:1.18}h2{margin-top:48px;line-height:1.3}
h3{margin-top:32px}p,li{max-width:88ch}a{color:#006777;text-underline-offset:3px}
a:focus-visible{outline:3px solid #cf7500;outline-offset:4px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 ui-monospace,monospace;
padding:16px;background:#eef3f2;border-radius:6px}
code{font-family:ui-monospace,monospace;font-size:.88em;overflow-wrap:anywhere}
table{border-collapse:collapse;font:14px/1.55 system-ui,sans-serif;max-width:100%}
td,th{border-bottom:1px solid #c7d2d3;padding:9px;text-align:left}
th{font-weight:650;background:#edf3f3}.output{overflow-x:auto;margin:12px 0 26px}
.cell{margin:28px 0;border-left:3px solid #087e8b;padding-left:16px}
.cell-label{font-size:12px;font-weight:650;color:#06636e;margin-bottom:4px}
img{display:block;max-width:100%;height:auto}.equation{overflow-x:auto;margin:26px 0}
math{font-family:"STIX Two Math","Cambria Math",math}
nav,footer{font-size:14px;color:#45626c}footer{margin-top:48px}
@media(max-width:640px){main{padding:0 16px 40px}body{font-size:16px}h1{font-size:1.8rem}
table{font-size:12px}.cell{padding-left:8px}pre{padding:10px}}
""" + fmt.get_style_defs(".highlight")
document = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>HLE scoring: an accessible Python walkthrough</title><style>' + css +
    '</style></head><body><main><nav>'
    '<a href="https://benchmarkanatomy.substack.com/p/i-re-scored-humanitys-last-exam-llama">Article</a>'
    ' · <a href="hle_scoring_walkthrough.ipynb">Python notebook</a></nav>' +
    "".join(parts) +
    '<footer>Prepared October 19, 2026. Execution scope and dependency versions are recorded '
    'in notebook_verification.json.</footer></main></body></html>'
)
(HERE / "notebook.html").write_text(document)
print(f"Rendered {figure} embedded figures and {document.count('<math ')} equations.")
