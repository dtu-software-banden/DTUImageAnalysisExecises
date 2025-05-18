from utils.io_utils import load_image
from utils.morph_utils import histogram_stretch
from utils.plot_utils import plot_image


img = load_image("exams/2023-fall-Thor/section1/ardeche_river.jpg",grayscale=True)

stretched = histogram_stretch(img,min=0.2,max=0.8)

plot_image(img)
plot_image(stretched)