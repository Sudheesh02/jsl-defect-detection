"""
Download curated real images and annotations from NEU-DET and SteelDefectX benchmarks.
"""
import os
import requests

def download_samples():
    img_dir = os.path.join("sample_data", "raw_images")
    ann_dir = os.path.join("sample_data", "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # 1. NEU-DET Benchmark images & annotations
    base_img_url = "https://raw.githubusercontent.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection/master/IMAGES/"
    base_ann_url = "https://raw.githubusercontent.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection/master/ANNOTATIONS/"
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    sample_indices = [1, 2, 3, 5, 10, 25]

    for cls in classes:
        for i in sample_indices:
            img_name = f"{cls}_{i}.jpg"
            ann_name = f"{cls}_{i}.xml"
            
            img_dest = os.path.join(img_dir, img_name)
            if not os.path.exists(img_dest):
                r = requests.get(base_img_url + img_name, timeout=10)
                if r.status_code == 200:
                    with open(img_dest, "wb") as f:
                        f.write(r.content)
            
            ann_dest = os.path.join(ann_dir, ann_name)
            if not os.path.exists(ann_dest):
                r = requests.get(base_ann_url + ann_name, timeout=10)
                if r.status_code == 200:
                    with open(ann_dest, "wb") as f:
                        f.write(r.content)

    # 2. SteelDefectX sample images & masks (Bright scratch, Crease, Roll printing, Oil spot, Waist folding)
    steelx_samples = [
        "bs_05.jpg", "cr_01.jpg", "frp_01.jpg", "os_01.jpg", "wf_01.jpg", "in_01.jpg"
    ]
    steelx_base = "https://huggingface.co/datasets/Zhaosxian/SteelDefectX/resolve/main/val/"
    for s_name in steelx_samples:
        dest = os.path.join(img_dir, f"steelx_{s_name}")
        if not os.path.exists(dest):
            try:
                r = requests.get(steelx_base + s_name, timeout=10)
                if r.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(r.content)
            except Exception as e:
                print(f"Error downloading {s_name}: {e}")

    # 3. Create clean/defect-free strip reference images by synthetic uniform rolling noise or bilateral filtered steel strip
    clean_dest = os.path.join(img_dir, "defect_free_strip_1.jpg")
    if not os.path.exists(clean_dest):
        import numpy as np
        from PIL import Image
        # High quality brushed stainless steel grain texture without defects
        h, w = 200, 200
        base_gray = 128
        # Horizontal brushed texture
        grain = np.random.normal(0, 4, (h, 1)).repeat(w, axis=1)
        noise = np.random.normal(0, 2, (h, w))
        steel_tex = np.clip(base_gray + grain + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(steel_tex)
        img.save(clean_dest)

    clean_dest2 = os.path.join(img_dir, "defect_free_strip_2.jpg")
    if not os.path.exists(clean_dest2):
        import numpy as np
        from PIL import Image
        h, w = 200, 200
        base_gray = 145
        grain = np.random.normal(0, 3, (h, 1)).repeat(w, axis=1)
        noise = np.random.normal(0, 2.5, (h, w))
        steel_tex = np.clip(base_gray + grain + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(steel_tex)
        img.save(clean_dest2)

    total_imgs = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Dataset preparation complete! Total test images available: {total_imgs}")

if __name__ == "__main__":
    download_samples()
