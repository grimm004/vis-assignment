# #!/usr/bin/env python
#
# # This is a simple volume rendering example that uses a
# # vtkGPUVolumeRayCastMapper
#
# # noinspection PyUnresolvedReferences
# import numpy as np
# import vtkmodules.vtkInteractionStyle
# # noinspection PyUnresolvedReferences
# import vtkmodules.vtkRenderingOpenGL2
# from PIL import Image
# from vtkmodules.util.misc import vtkGetDataRoot
# from vtkmodules.vtkCommonColor import vtkNamedColors
# from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR
# from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
# from vtkmodules.vtkFiltersSources import vtkCylinderSource
# from vtkmodules.vtkIOLegacy import vtkStructuredPointsReader
# from vtkmodules.vtkRenderingCore import (
#     vtkActor,
#     vtkVolume,
#     vtkPolyDataMapper,
#     vtkRenderWindow,
#     vtkRenderWindowInteractor,
#     vtkRenderer, vtkImageMapper, vtkActor2D, vtkColorTransferFunction, vtkVolumeProperty
# )
# from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
#
# import vtk
#
#
# VTK_DATA_ROOT = vtkGetDataRoot()
# print(VTK_DATA_ROOT)
#
# # Create the standard renderer, render window and interactor
# ren = vtkRenderer()
# renWin = vtkRenderWindow()
# renWin.AddRenderer(ren)
# iren = vtkRenderWindowInteractor()
# iren.SetRenderWindow(renWin)
#
# # Create the reader for the data
# reader = vtkStructuredPointsReader()
# reader.SetFileName("./Data/ironProt.vtk")
#
# # Create transfer mapping scalar value to opacity
# opacityTransferFunction = vtkPiecewiseFunction()
# opacityTransferFunction.AddPoint(20, 0.0)
# opacityTransferFunction.AddPoint(255, 0.2)
#
# # Create transfer mapping scalar value to color
# colorTransferFunction = vtkColorTransferFunction()
# colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
# colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
# colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
# colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
# colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)
#
# # The property describes how the data will look
# volumeProperty = vtkVolumeProperty()
# volumeProperty.SetColor(colorTransferFunction)
# volumeProperty.SetScalarOpacity(opacityTransferFunction)
# volumeProperty.ShadeOn()
# # volumeProperty.SetInterpolationTypeToLinear()
#
# with Image.open("./jwst-image-depth.png") as image:
#     image: Image = image.convert("RGB")
#     image_array: np.ndarray = np.array(image)
#
# image_data = vtkImageData()
# image_data.SetDimensions(image_array.shape[0], image_array.shape[1], 1)
# image_data.AllocateScalars(VTK_UNSIGNED_CHAR, 3)
# for x in range(0, image_array.shape[0]):
#     for y in range(0, image_array.shape[1]):
#         for c in range(0, 3):
#             # pixel = image_data.GetScalarPointer(x, y, 0)
#             image_data.SetScalarComponentFromFloat(x, y, 0, c, image_array[x, y, c])
#             # pixel = np.array(image_array[x, y, :])
#
# # The mapper / ray cast function know how to render the data
# volumeMapper = vtkGPUVolumeRayCastMapper()
# volumeMapper.SetBlendModeToComposite()
# # volumeMapper.SetInputConnection(reader.GetOutputPort())
# volumeMapper.SetInputData(image_data)
#
# # The volume holds the mapper and the property and
# # can be used to position/orient the volume
# volume = vtkVolume()
# volume.SetMapper(volumeMapper)
# volume.SetProperty(volumeProperty)
#
# ren.AddVolume(volume)
# ren.SetBackground(1, 1, 1)
# renWin.SetSize(600, 600)
# renWin.Render()
#
#
# def CheckAbort(obj, event):
#     if obj.GetEventPending() != 0:
#         obj.SetAbortRender(1)
#
#
# renWin.AddObserver("AbortCheckEvent", CheckAbort)
#
# iren.Initialize()
# renWin.Render()
# iren.Start()


#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkPointSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def main():
    colors = vtkNamedColors()

    # create a rendering window and renderer
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetWindowName('PointSource')

    # create a renderwindowinteractor
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create a point cloud
    src = vtkPointSource()
    src.SetCenter(0, 0, 0)
    src.SetNumberOfPoints(50)
    src.SetRadius(5)
    src.Update()

    # mapper
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(src.GetOutputPort())

    # actor
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('Tomato'))
    actor.GetProperty().SetPointSize(4)

    # assign actor to the renderer
    ren.AddActor(actor)
    ren.SetBackground(colors.GetColor3d('DarkGreen'))

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()


if __name__ == '__main__':
    main()
