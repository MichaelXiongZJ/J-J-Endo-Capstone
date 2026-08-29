"""Convert USER_GUIDE.md to a beautifully styled PDF using headless Edge/Chrome."""
import os
import pathlib
import subprocess
import markdown

edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
browser_exe = next((p for p in edge_paths if os.path.exists(p)), None)
if not browser_exe:
    raise RuntimeError("Neither Microsoft Edge nor Google Chrome found for PDF export.")

md_file = pathlib.Path("USER_GUIDE.md").resolve()
html_file = pathlib.Path("USER_GUIDE.html").resolve()
pdf_file = pathlib.Path("USER_GUIDE.pdf").resolve()

md_content = md_file.read_text(encoding="utf-8")
html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

css = """
@page {
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #24292f;
    margin: 0;
    padding: 0;
}
h1 {
    font-size: 20pt;
    border-bottom: 2px solid #eaecef;
    padding-bottom: 8px;
    margin-top: 0;
}
h2 {
    font-size: 14pt;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 5px;
    margin-top: 24px;
    page-break-after: avoid;
}
h3 {
    font-size: 11pt;
    margin-top: 16px;
    page-break-after: avoid;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #d0d7de;
    padding: 6px 10px;
    text-align: left;
}
th {
    background-color: #f6f8fa;
    font-weight: 600;
}
tr:nth-child(even) {
    background-color: #fcfcfc;
}
code {
    background-color: #f6f8fa;
    padding: 2px 4px;
    border-radius: 4px;
    font-size: 8.5pt;
    font-family: Consolas, "Liberation Mono", Menlo, Courier, monospace;
}
pre {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    padding: 10px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 8pt;
    font-family: Consolas, "Liberation Mono", Menlo, Courier, monospace;
    page-break-inside: avoid;
}
pre code {
    background: transparent;
    padding: 0;
}
blockquote {
    margin: 10px 0;
    padding: 0 1em;
    color: #57606a;
    border-left: 0.25em solid #d0d7de;
}
hr {
    border: none;
    border-top: 1px solid #eaecef;
    margin: 20px 0;
}
"""

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>User Guide — Warehouse Safety CV (J&J Capstone)</title>
<style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

html_file.write_text(full_html, encoding="utf-8")

cmd = [
    browser_exe,
    "--headless=new",
    "--disable-gpu",
    f"--print-to-pdf={str(pdf_file)}",
    "--no-pdf-header-footer",
    str(html_file),
]

subprocess.run(cmd, check=True)

# Clean up temp HTML file
if html_file.exists():
    html_file.unlink()

print(f"Generated {pdf_file} ({pdf_file.stat().st_size:,} bytes)")
