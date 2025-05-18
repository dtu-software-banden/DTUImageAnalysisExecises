import SimpleITK as sitk
from utils.io_utils import load_image
from utils.plot_utils import imshow_orthogonal_view, plot_image

fixed_image = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v1.nii")
moving_image = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v2.nii")




# Step 1: Read or assume your images
# fixed_image = sitk.ReadImage("fixed.nii.gz")
# moving_image = sitk.ReadImage("moving.nii.gz")

# Step 2: Initial transform - Euler3D (rigid: rotation + translation)
initial_transform = sitk.CenteredTransformInitializer(
    fixed_image,
    moving_image,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY  # or MOMENTS
)

# Step 3: Set up registration method
registration_method = sitk.ImageRegistrationMethod()

# Similarity metric
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

# Interpolator
registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer
registration_method.SetOptimizerAsRegularStepGradientDescent(
    learningRate=2.0,
    minStep=1e-4,
    numberOfIterations=200,
    gradientMagnitudeTolerance=1e-8
)

# Setup for multi-resolution framework
registration_method.SetShrinkFactorsPerLevel([4,2,1])
registration_method.SetSmoothingSigmasPerLevel([2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Set initial transform
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# Step 4: Run registration
final_transform = registration_method.Execute(fixed_image, moving_image)

# Step 5: Apply the transform to align the moving image
resampled_image = sitk.Resample(moving_image, fixed_image, final_transform,
                                sitk.sitkLinear, 0.0, moving_image.GetPixelID())

# Step 6: Save or visualize
# sitk.WriteImage(resampled_image, "registered_image.nii.gz")
# sitk.WriteTransform(final_transform, "rigid_transform.tfm")

# array1 = sitk.GetArrayFromImage(resampled_image)
# array2 = sitk.GetArrayFromImage(final_transform)

# imshow_orthogonal_view(resampled_image)
# # imshow_orthogonal_view(array2)
# imshow_orthogonal_view(final_transform)

print("Final transform:",final_transform)
