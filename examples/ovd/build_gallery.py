import argparse
import html
import json
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a switchable HTML gallery from VLM-R1 OVD outputs.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="TITLE=OUTPUT_DIR",
        help="Gallery case title and output directory. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/ovd_gallery/index.html"))
    parser.add_argument("--title", default="VLM-R1 OVD Example Gallery")
    return parser.parse_args()


def find_input_image(output_dir):
    for path in sorted(output_dir.iterdir()):
        if path.name == "annotated.png":
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    raise FileNotFoundError(f"No input image found in {output_dir}")


def read_detections(output_dir):
    detections_path = output_dir / "detections.json"
    if not detections_path.exists():
        return []
    return json.loads(detections_path.read_text(encoding="utf-8"))


def infer_labels(detections):
    labels = []
    for item in detections:
        label = item.get("label")
        if label and label not in labels:
            labels.append(label)
    return ", ".join(labels) if labels else "none"


def to_posix_relative(path, start):
    try:
        return Path(os.path.relpath(Path(path).resolve(), start.resolve().parent)).as_posix()
    except ValueError:
        return Path(path).resolve().as_uri()


def parse_case(value):
    if "=" not in value:
        raise ValueError(f"--case must use TITLE=OUTPUT_DIR, got: {value}")
    title, output_dir = value.split("=", 1)
    title = title.strip()
    if not title:
        raise ValueError(f"--case title cannot be empty: {value}")
    return title, Path(output_dir)


def load_cases(case_args, gallery_output):
    cases = []
    for raw_case in case_args:
        title, output_dir = parse_case(raw_case)
        output_dir = output_dir.resolve()
        annotated_path = output_dir / "annotated.png"
        report_path = output_dir / "report.html"
        if not annotated_path.exists():
            raise FileNotFoundError(f"Missing annotated output: {annotated_path}")
        if not report_path.exists():
            raise FileNotFoundError(f"Missing report: {report_path}")

        input_path = find_input_image(output_dir)
        detections = read_detections(output_dir)
        cases.append(
            {
                "title": title,
                "labels": infer_labels(detections),
                "input": to_posix_relative(input_path, gallery_output),
                "annotated": to_posix_relative(annotated_path, gallery_output),
                "report": to_posix_relative(report_path, gallery_output),
                "detections": json.dumps(detections, indent=2),
            }
        )
    return cases


def write_gallery(cases, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases_json = json.dumps(cases)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; background: #f6f7fb; color: #172033; font-family: Arial, sans-serif; }}
main {{ max-width: 1260px; margin: 28px auto; padding: 0 18px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
.subtitle {{ margin: 0 0 22px; color: #475569; }}
.card {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06); }}
.top {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 14px; }}
.case-title {{ margin: 0 0 6px; font-size: 22px; }}
.labels {{ margin: 0; color: #334155; }}
.buttons {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; max-width: 620px; }}
button {{ border: 1px solid #cbd5e1; border-radius: 6px; background: white; padding: 8px 12px; font-size: 14px; cursor: pointer; }}
button.active {{ background: #172033; color: white; border-color: #172033; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.panel {{ background: #eef2f7; border: 1px solid #d8dee9; border-radius: 6px; padding: 10px; }}
.panel h3 {{ margin: 0 0 8px; font-size: 16px; color: #475569; }}
.frame {{ height: 650px; background: white; border-radius: 5px; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
.frame img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
.links {{ margin: 14px 0; }}
.links a {{ color: #1d4ed8; text-decoration: none; font-weight: 700; }}
pre {{ max-height: 300px; overflow: auto; white-space: pre-wrap; background: #f1f5f9; border-radius: 6px; padding: 12px; }}
@media (max-width: 900px) {{ .top, .grid {{ display: block; }} .buttons {{ justify-content: flex-start; margin-top: 12px; }} .panel + .panel {{ margin-top: 14px; }} .frame {{ height: 520px; }} }}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p class="subtitle">Local inference outputs generated by <code>examples/ovd/infer_ovd.py</code>.</p>
<section class="card">
  <div class="top">
    <div>
      <h2 class="case-title" id="caseTitle"></h2>
      <p class="labels" id="caseLabels"></p>
    </div>
    <div class="buttons" id="caseButtons"></div>
  </div>
  <div class="grid">
    <section class="panel"><h3>Input</h3><div class="frame"><img id="inputImage" alt="Input image" /></div></section>
    <section class="panel"><h3>Annotated Output</h3><div class="frame"><img id="annotatedImage" alt="Annotated output" /></div></section>
  </div>
  <div class="links"><a id="reportLink" href="#">Open report</a></div>
  <pre id="detections"></pre>
</section>
</main>
<script>
const cases = {cases_json};
const buttons = document.getElementById('caseButtons');
function selectCase(index) {{
  const item = cases[index];
  document.getElementById('caseTitle').textContent = item.title;
  document.getElementById('caseLabels').textContent = `Labels: ${{item.labels}}`;
  document.getElementById('inputImage').src = item.input;
  document.getElementById('annotatedImage').src = item.annotated;
  document.getElementById('reportLink').href = item.report;
  document.getElementById('detections').textContent = item.detections;
  [...buttons.children].forEach((button, buttonIndex) => button.classList.toggle('active', buttonIndex === index));
}}
cases.forEach((item, index) => {{
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = item.title;
  button.addEventListener('click', () => selectCase(index));
  buttons.appendChild(button);
}});
if (cases.length) {{
  selectCase(0);
}}
</script>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")


def main():
    args = parse_args()
    if not args.case:
        raise SystemExit("Pass at least one --case TITLE=OUTPUT_DIR.")
    cases = load_cases(args.case, args.output)
    write_gallery(cases, args.output, args.title)
    print(f"Wrote gallery: {args.output}")


if __name__ == "__main__":
    main()
