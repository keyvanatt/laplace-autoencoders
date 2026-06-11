import json, base64, os

with open('notebooks/eval_checkpoints.ipynb','r',encoding='utf-8') as f:
    nb = json.load(f)

out_dir = 'article/images'
os.makedirs(out_dir, exist_ok=True)

img_count = 0
for i, cell in enumerate(nb['cells']):
    for j, output in enumerate(cell.get('outputs', [])):
        if output.get('output_type') in ('display_data','execute_result'):
            data = output.get('data', {})
            for fmt, content in data.items():
                if 'png' in fmt or 'jpeg' in fmt:
                    ext = 'png' if 'png' in fmt else 'jpg'
                    raw = content if isinstance(content, str) else ''.join(content)
                    img_bytes = base64.b64decode(raw)
                    fname = f'{out_dir}/nb_cell{i:02d}_{img_count:02d}.{ext}'
                    with open(fname,'wb') as fout:
                        fout.write(img_bytes)
                    print(f'  saved {fname}  ({len(img_bytes)//1024}KB)')
                    img_count += 1

print(f'Total: {img_count} images')