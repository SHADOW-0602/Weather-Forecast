from pathlib import Path

from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
pages = sorted(
    (root / "report_output" / "qa_pages").glob("page-*.png"),
    key=lambda p: int(p.stem.split("-")[-1]),
)
out_dir = root / "report_output" / "qa_sheets"
out_dir.mkdir(parents=True, exist_ok=True)

thumb_w = 420
for group_start in range(0, len(pages), 5):
    selected = pages[group_start:group_start + 5]
    thumbs = []
    for page in selected:
        im = Image.open(page).convert("RGB")
        h = round(im.height * thumb_w / im.width)
        thumbs.append(im.resize((thumb_w, h)))
    cell_h = max(im.height for im in thumbs) + 34
    sheet = Image.new("RGB", (thumb_w * len(thumbs), cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (page, im) in enumerate(zip(selected, thumbs)):
        x = i * thumb_w
        sheet.paste(im, (x, 28))
        draw.text((x + 8, 7), f"Page {int(page.stem.split('-')[-1])}", fill="black")
    first_num = group_start + 1
    last_num = group_start + len(selected)
    sheet.save(out_dir / f"pages_{first_num:02d}_{last_num:02d}.png")

print(f"Created {(len(pages) + 4) // 5} contact sheets")
