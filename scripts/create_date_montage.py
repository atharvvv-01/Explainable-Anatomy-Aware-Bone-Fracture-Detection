import os
from PIL import Image, ImageDraw, ImageFont

dates = ['2026-05-19', '2026-05-21', '2026-05-25', '2026-05-29', '2026-09-06', '2026-09-28']
os.makedirs('reports/visualizations', exist_ok=True)

# For each date, pick 3 fracture and 3 non-fracture images, resize to 200x200 and stitch together
canvas = Image.new('RGB', (6 * 200, len(dates) * 220), (30, 30, 30))
draw = ImageDraw.Draw(canvas)

for row_idx, d in enumerate(dates):
    frac_files = [f for f in os.listdir('Dataset/Dataset/Fracture') if d in f][:3]
    nf_files = [f for f in os.listdir('Dataset/Dataset/Non fracture') if d in f][:3]
    
    samples = [(f, 'Dataset/Dataset/Fracture', 'Frac') for f in frac_files] + \
              [(f, 'Dataset/Dataset/Non fracture', 'Norm') for f in nf_files]
              
    for col_idx, (fname, folder, label) in enumerate(samples):
        img_path = os.path.join(folder, fname)
        try:
            with Image.open(img_path) as im:
                im_rgb = im.convert('RGB')
                im_thumb = im_rgb.resize((190, 190))
                x = col_idx * 200 + 5
                y = row_idx * 220 + 25
                canvas.paste(im_thumb, (x, y))
                draw.text((x, y - 20), f"{d} | {label}", fill=(255, 255, 255))
        except Exception as e:
            print(f"Error {fname}: {e}")

canvas.save('reports/visualizations/date_samples_montage.jpg')
print("Saved reports/visualizations/date_samples_montage.jpg")
