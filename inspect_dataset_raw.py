import os
import hashlib
from PIL import Image

def analyze_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(f"Analyzing {folder_path}: {len(files)} files")
    corrupted = []
    formats = {}
    modes = {}
    sizes = []
    hashes = {}
    duplicates = []
    
    for f in files:
        p = os.path.join(folder_path, f)
        try:
            with open(p, 'rb') as fp:
                data = fp.read()
                h = hashlib.sha256(data).hexdigest()
                if h in hashes:
                    duplicates.append((f, hashes[h]))
                else:
                    hashes[h] = f
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                fmt = img.format
                mode = img.mode
                size = img.size
                formats[fmt] = formats.get(fmt, 0) + 1
                modes[mode] = modes.get(mode, 0) + 1
                sizes.append(size)
        except Exception as e:
            corrupted.append((f, str(e)))
            
    return {
        'total': len(files),
        'corrupted': corrupted,
        'formats': formats,
        'modes': modes,
        'sizes_count': len(sizes),
        'min_size': min(sizes) if sizes else None,
        'max_size': max(sizes) if sizes else None,
        'duplicates': duplicates,
        'hashes': hashes
    }

frac = analyze_folder('Dataset/Dataset/Fracture')
non_frac = analyze_folder('Dataset/Dataset/Non fracture')

print('\n--- FRACTURE SUMMARY ---')
print(f"Total: {frac['total']}")
print(f"Corrupted: {len(frac['corrupted'])}")
print(f"Duplicates within Fracture: {len(frac['duplicates'])}")
print(f"Formats: {frac['formats']}")
print(f"Modes: {frac['modes']}")
print(f"Min Size: {frac['min_size']}, Max Size: {frac['max_size']}")

print('\n--- NON-FRACTURE SUMMARY ---')
print(f"Total: {non_frac['total']}")
print(f"Corrupted: {len(non_frac['corrupted'])}")
print(f"Duplicates within Non-fracture: {len(non_frac['duplicates'])}")
print(f"Formats: {non_frac['formats']}")
print(f"Modes: {non_frac['modes']}")
print(f"Min Size: {non_frac['min_size']}, Max Size: {non_frac['max_size']}")

cross_duplicates = set(frac['hashes'].keys()).intersection(set(non_frac['hashes'].keys()))
print(f"\nCross-class exact duplicates (in both Fracture and Non-fracture): {len(cross_duplicates)}")
for h in cross_duplicates:
    print(f"  Frac: {frac['hashes'][h]} <===> Non-Frac: {non_frac['hashes'][h]}")
