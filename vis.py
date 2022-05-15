from typing import Tuple, List
from argparse import ArgumentParser, BooleanOptionalAction
from PIL import Image
from matplotlib.colors import Colormap
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from random import randint, random
import numpy as np
from tqdm import tqdm
import sys

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkVolume,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer, vtkImageMapper, vtkActor2D
)

IM_WIDTH = 2048
IM_HEIGHT = 2048
IM_BG_COLOR = "#000000"
IM_FILE = "./jwst-image.png"
GALAXY_IMG_FILE = "./galaxy-icon-0.png"
GALAXY_COLORMAP = "rainbow"
GALAXY_MAX_WIDTH = 64
GALAXY_MAX_HEIGHT = 64
GALAXY_MAX_COUNT = 10000
GALAXY_MAX_TRIES = 5


def hex_to_colour(hex_string: str) -> Tuple[int, ...]:
    return tuple(map(int, bytes.fromhex(hex_string.lstrip("#"))))[:3]


def get_random_color(colormap: Colormap) -> Tuple[int, ...]:
    return tuple(map(lambda x: int(255.0 * x), colormap(random())))


class BoundingBox:
    def __init__(self, pos: Tuple[int, int] = (0, 0), size: Tuple[int, int] = (1, 1)):
        self.x, self.y = pos
        self.width, self.height = size

    @property
    def left(self) -> int:
        return self.x

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def top(self) -> int:
        return self.y

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    @size.setter
    def size(self, new_size: Tuple[int, int]) -> None:
        self.width, self.height = new_size

    @property
    def pos(self) -> Tuple[int, int]:
        return self.x, self.y

    @pos.setter
    def pos(self, new_size: Tuple[int, int]) -> None:
        self.x, self.y = new_size

    def intersects(self, bb: "BoundingBox") -> bool:
        return \
            bb.right > self.left and bb.left < self.right and \
            bb.bottom > self.top and bb.top < self.bottom


def main_simulation(
    im_size: Tuple[int, int],
    im_bg_color: str,
    im_file: str,
    galaxy_colormap: str,
    galaxy_img_file: str,
    galaxy_max_width: int,
    galaxy_max_height: int,
    galaxy_max_count: int,
    galaxy_max_tries: int
) -> int:
    print("Generating simulated image...", end="")

    colormap: Colormap = cm.get_cmap(galaxy_colormap)
    image: Image = Image.new("RGB", im_size, hex_to_colour(im_bg_color))

    with Image.open(galaxy_img_file) as galaxy_image:
        galaxy_image: Image = galaxy_image.split()[-1]

        bounding_boxes: List[BoundingBox] = []

        for _ in tqdm(range(galaxy_max_count)):
            for _ in range(galaxy_max_tries):
                proposed_size = \
                    randint(1, galaxy_max_width), \
                    randint(1, galaxy_max_height)

                transformed_galaxy: Image = galaxy_image \
                    .resize(proposed_size, Image.ANTIALIAS) \
                    .rotate(randint(0, 180), expand=True)
                proposed_size = transformed_galaxy.size

                proposed_pos = \
                    randint(0, IM_WIDTH - transformed_galaxy.size[0]), \
                    randint(0, IM_HEIGHT - transformed_galaxy.size[1])

                proposed_bounding_box = BoundingBox(proposed_pos, proposed_size)
                proposed_bounding_box.size = transformed_galaxy.size

                if any(proposed_bounding_box.intersects(bounding_box) for bounding_box in bounding_boxes):
                    continue

                colored_image: Image = Image.new("RGBA", transformed_galaxy.size, get_random_color(colormap))

                image.paste(colored_image, proposed_bounding_box.pos, mask=transformed_galaxy)
                bounding_boxes.append(proposed_bounding_box)

    print(" Writing output file...", end="")

    image.save(im_file)

    print(" Done")

    return 0


def depth_function(data: np.ndarray, invert: bool = False) -> np.ndarray:
    return 1.0 - np.sqrt(1.0 - data) if invert else np.sqrt(1.0 - data)


def plot_colourmap(colormap: Colormap, correct_depth: bool, seg_count: int = 1000, save: bool = True):
    xs = np.linspace(0.0, 1.0, seg_count)
    ys = colormap(xs)
    plt.figure(figsize=(8, 8), dpi=80)
    if correct_depth:
        plt.plot(xs, depth_function(ys[:, 2]), "b")
    else:
        plt.plot(xs, ys[:, 0], "r", xs, ys[:, 1], "g", xs, ys[:, 2], "b")
    if save:
        plt.savefig("depth-function.png" if correct_depth else "colourmap.png")
    else:
        plt.show()
    plt.clf()


def load_depths(image: Image) -> np.ndarray:
    return depth_function(np.array(image).astype(float)[:, :, 2] / 255.0)


def load_galaxy_mask(image: Image) -> np.ndarray:
    return np.any(np.array(image)[:, :], axis=2)


def main_vis(im_file: str, galaxy_colormap: Colormap) -> int:
    print("Starting visualisation...", end="")

    plot_colourmap(cm.get_cmap(galaxy_colormap), False)
    plot_colourmap(cm.get_cmap(galaxy_colormap), True)

    with Image.open(im_file) as image:
        image: Image = image.convert("RGB")

        image_array: np.ndarray = np.array(image)

        depths = load_depths(image) * 255.0
        galaxy_mask = load_galaxy_mask(image)

        Image.fromarray(depths.astype(np.uint8)).save("jwst-image-depth.png")
        Image.fromarray(galaxy_mask).save("jwst-image-galaxy-mask.png")

    colors = vtkNamedColors()
    # Set the background color.
    bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
    colors.SetColor("BkgColor", *bkg)

    # This creates a polygonal cylinder model with eight circumferential
    # facets.
    cylinder = vtkCylinderSource()
    cylinder.SetResolution(8)

    # The mapper is responsible for pushing the geometry into the graphics
    # library. It may also do color mapping, if scalars or other
    # attributes are defined.
    cylinder_mapper = vtkPolyDataMapper()
    # noinspection PyArgumentList
    cylinder_mapper.SetInputConnection(cylinder.GetOutputPort(0))

    # The actor is a grouping mechanism: besides the geometry (mapper), it
    # also has a property, transformation matrix, and/or texture map.
    # Here we set its color and rotate it -22.5 degrees.
    cylinder_actor = vtkActor()
    cylinder_actor.SetMapper(cylinder_mapper)
    cylinder_actor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
    cylinder_actor.RotateX(30.0)
    cylinder_actor.RotateY(-45.0)

    # Create the graphics structure. The renderer renders into the render
    # window. The render window interactor captures mouse events and will
    # perform appropriate camera or actor manipulation depending on the
    # nature of the events.
    ren = vtkRenderer()

    img_mapper = vtkImageMapper()
    img_actor = vtkActor2D()
    image_data = vtkImageData()
    image_data.SetDimensions(image_array.shape[0], image_array.shape[1], 1)
    image_data.AllocateScalars(VTK_UNSIGNED_CHAR, 3)
    for x in range(0, image_array.shape[0]):
        for y in range(0, image_array.shape[1]):
            for c in range(0, 3):
                # pixel = image_data.GetScalarPointer(x, y, 0)
                image_data.SetScalarComponentFromFloat(x, y, 0, c, image_array[x, y, c])
                # pixel = np.array(image_array[x, y, :])
    img_mapper.SetInputData(image_data)
    img_mapper.SetColorWindow(255)
    img_mapper.SetColorLevel(127.5)
    img_actor.SetMapper(img_mapper)
    ren.AddActor(img_actor)

    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(ren)
    # noinspection PyArgumentList
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    # Add the actors to the renderer, set the background and size
    ren.AddActor(cylinder_actor)
    # noinspection PyArgumentList
    ren.SetBackground(colors.GetColor3d("BkgColor"))
    ren_win.SetSize(300, 300)
    ren_win.SetWindowName('CylinderExample')

    # This allows the interactor to initialise itself. It has to be
    # called before an event loop.
    iren.Initialize()

    # We'll zoom in a little by accessing the camera and invoking a "Zoom"
    # method on it.
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1.5)
    ren_win.Render()

    print(" Done")

    # Start the event loop.
    iren.Start()

    return 0


def main() -> int:
    parser = ArgumentParser(description="Generate and visualise simulated data from the JWST.")

    # Data simulation arguments
    parser.add_argument(
        "--width",
        dest="im_width",
        type=int,
        default=IM_WIDTH,
        help="Width of simulated image in pixels."
    )
    parser.add_argument(
        "--height",
        dest="im_height",
        type=int,
        default=IM_HEIGHT,
        help="Height of simulated image in pixels."
    )
    parser.add_argument(
        "--bg-colour",
        dest="im_bg_color",
        type=str,
        default=IM_BG_COLOR,
        help="Background colour of simulated image."
    )
    parser.add_argument(
        "-o", "--im-file",
        dest="im_file",
        type=str,
        default=IM_FILE,
        help="Simulated image file output path."
    )
    parser.add_argument(
        "--galaxy-file",
        dest="galaxy_img_file",
        type=str,
        default=GALAXY_IMG_FILE,
        help="Path to galaxy image file."
    )
    parser.add_argument(
        "--galaxy-colourmap",
        dest="galaxy_colormap",
        type=str,
        default=GALAXY_COLORMAP,
        help="Colormap for use in random galaxy generation."
    )
    parser.add_argument(
        "--galaxy-max-width",
        dest="galaxy_max_width",
        type=int,
        default=GALAXY_MAX_WIDTH,
        help="Maximum width of galaxies in pixels."
    )
    parser.add_argument(
        "--galaxy-max-height",
        dest="galaxy_max_height",
        type=int,
        default=GALAXY_MAX_HEIGHT,
        help="Maximum height of galaxies in pixels."
    )
    parser.add_argument(
        "--galaxy-max-count",
        dest="galaxy_max_count",
        type=int,
        default=GALAXY_MAX_COUNT,
        help="Maximum number of galaxies in simulated image."
    )
    parser.add_argument(
        "--galaxy-max-tries",
        dest="galaxy_max_tries",
        type=int,
        default=GALAXY_MAX_TRIES,
        help="Maximum number of galaxy placement attempts."
    )
    parser.add_argument(
        "-s",
        "--sim",
        dest="sim",
        default=False,
        action=BooleanOptionalAction,
        help="Run the JWST image simulation."
    )
    parser.add_argument(
        "-v",
        "--vis",
        dest="vis",
        default=False,
        action=BooleanOptionalAction,
        help="Run the JWST image data visualisation."
    )

    args = parser.parse_args()

    if args.sim and (err_code := main_simulation(
        (args.im_width, args.im_height),
        args.im_bg_color,
        args.im_file,
        args.galaxy_colormap,
        args.galaxy_img_file,
        args.galaxy_max_width,
        args.galaxy_max_height,
        args.galaxy_max_count,
        args.galaxy_max_tries
    )):
        return err_code

    if args.vis and (err_code := main_vis(args.im_file, args.galaxy_colormap)):
        return err_code

    return 0


if __name__ == "__main__":
    sys.exit(main())
