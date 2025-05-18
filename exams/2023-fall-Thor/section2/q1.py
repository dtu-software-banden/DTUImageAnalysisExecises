from utils.io_utils import load_image
import SimpleITK as sitk

from utils.plot_utils import imshow_orthogonal_view, plot_image

sitkImage1 = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v1.nii")
sitkImage2 = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v2.nii")




matrix = [0.98, -0.16, 0.17,
          0.26, 0.97, 0.0,
         -0.17, 0.04, 0.98]
translation = [0, -15, 0]


center = sitkImage2.TransformContinuousIndexToPhysicalPoint([
    sz/2.0 for sz in sitkImage2.GetSize()
])

transform = sitk.AffineTransform(3)

transform.SetMatrix(matrix)
transform.SetTranslation(translation)
transform.SetCenter(center) # Default is actually center. Does not change anything


resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(sitkImage1)       # Match size, spacing, origin, direction
resampler.SetInterpolator(sitk.sitkLinear)    # Or sitkNearestNeighbor if labels
resampler.SetTransform(transform)
resampler.SetDefaultPixelValue(0)

transformed2 = resampler.Execute(sitkImage2)

array1 = sitk.GetArrayFromImage(sitkImage1)
array2 = sitk.GetArrayFromImage(sitkImage2)
array2trans = sitk.GetArrayFromImage(transformed2)

imshow_orthogonal_view(array1)
# imshow_orthogonal_view(array2)
imshow_orthogonal_view(array2trans)