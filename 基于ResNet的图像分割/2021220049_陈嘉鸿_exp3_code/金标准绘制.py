import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from PIL import Image

class InteractivePolygon:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.mask = np.zeros(image.size[::-1], dtype=np.uint8)
        self.selector = PolygonSelector(ax, self.on_select, useblit=True)
        self.poly = None

    def on_select(self, verts):
        path = Path(verts)
        self.poly = path
        self.update_mask(path)
        self.ax.imshow(self.mask, cmap='gray', alpha=0.5)
        plt.draw()

    def update_mask(self, path):
        x, y = np.meshgrid(np.arange(self.mask.shape[1]), np.arange(self.mask.shape[0]))
        points = np.vstack((x.flatten(), y.flatten())).T
        grid = path.contains_points(points).reshape(self.mask.shape)
        self.mask[grid] = 1

    def get_mask(self):
        return self.mask

def main(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots()
    ax.imshow(image)
    poly_selector = InteractivePolygon(ax, image)
    plt.show()

    # 获取并保存生成的掩码
    mask = poly_selector.get_mask()
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(mask_path)

    print(f"金标准掩码已保存至: {mask_path}")

if __name__ == "__main__":
    image_path = "b1f672549b20c2ccba7305e8a230f9ab.jpeg"
    mask_path = "ground_truth_mask.png"
    main(image_path, mask_path)
