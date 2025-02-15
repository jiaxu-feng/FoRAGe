import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting, AutoPipelineForImage2Image
from rembg import remove, new_session
import requests
from io import BytesIO
from transparent_background import Remover
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM

def generate_rectangular_mask(transformed_mask, object_image):
    # Convert the mask image to a numpy array
    mask_array = np.array(transformed_mask)

    # Find the boundaries of non-zero pixels
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Calculate the aspect ratio of the object image
    object_width, object_height = object_image.size
    aspect_ratio = object_width / object_height

    # Calculate the width and height of the original mask area
    original_width = cmax - cmin
    original_height = rmax - rmin

    # If the aspect ratio of the original mask is less than that of the object image, extend the width, otherwise extend the height
    if original_width / original_height < aspect_ratio:
        new_width = int(original_height * aspect_ratio)  # Ensure the new width is greater than or equal to the original width
        new_height = original_height
    else:
        new_height = int(original_width / aspect_ratio)  # Ensure the new height is greater than or equal to the original height
        new_width = original_width

    # Calculate the center position of the new mask area
    center_x = (cmin + cmax) // 2
    center_y = (rmin + rmax) // 2

    # Calculate the boundaries of the new mask area, keeping the mask center position unchanged
    new_rmin = max(0, center_y - new_height // 2)
    new_rmax = min(mask_array.shape[0], center_y + new_height // 2)
    new_cmin = max(0, center_x - new_width // 2)
    new_cmax = min(mask_array.shape[1], center_x + new_width // 2)

    # Create the new mask area, ensuring the new mask is greater than or equal to the original mask
    new_mask_array = np.zeros_like(mask_array)
    new_mask_array[new_rmin:new_rmax, new_cmin:new_cmax] = 255

    # Convert the new mask array back to a PIL image
    new_mask_image = Image.fromarray(new_mask_array)
    return new_mask_image

def generate_square_mask(transformed_mask):
    # Convert the mask image to a numpy array
    mask_array = np.array(transformed_mask)

    # Find the boundaries of non-zero pixels
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Create a new rectangular mask
    new_mask_array = np.zeros_like(mask_array)
    new_mask_array[rmin:rmax+1, cmin:cmax+1] = 255

    # Convert the new mask array back to a PIL image
    new_mask_image = Image.fromarray(new_mask_array)
    return new_mask_image

# --------- Step 1: Generate Alpha Mask ---------
def generate_mask_from_png(png):
    img = np.array(png)
    if img is None:
        raise ValueError(f"Image at {png_path} could not be loaded.")
    
    if img.shape[2] == 4:
        mask = img[:, :, 3]
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        return mask
    else:
        raise ValueError('The image does not have an alpha channel.')

def resize_and_mask(source_image, target_image):
    img_a = cv2.imread(source_image)
    img_b = np.array(target_image)
    mask_b = generate_mask_from_png(target_image)

    if img_a is None or img_b is None or mask_b is None:
        raise ValueError("One or more images could not be loaded. Please check the paths.")

    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_b, descriptors_a, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([keypoints_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if len(good_matches) >= 4:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = mask_b.shape[:2]
        transformed_mask = cv2.warpPerspective(mask_b, M, (img_a.shape[1], img_a.shape[0]))

        return transformed_mask
    else:
        raise ValueError("Not enough matches found to compute the homography.")

# Generate replacement subject mask
def get_alpha_mask(image_path):
    image = image_path.convert("RGBA")
    alpha_channel = image.split()[-1]  # Get the Alpha channel (transparency)
    alpha_np = np.array(alpha_channel)
    
    mask = alpha_np > 0
    return mask

def generate_square_mask(transformed_mask, object_image):
    mask_array = np.array(transformed_mask)

    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    object_width, object_height = object_image.size
    aspect_ratio = object_width / object_height

    original_width = cmax - cmin
    original_height = rmax - rmin

    if original_width / original_height < aspect_ratio:
        new_width = int(original_height * aspect_ratio)
        new_height = original_height
    else:
        new_height = int(original_width / aspect_ratio)
        new_width = original_width

    center_x = (cmin + cmax) // 2
    center_y = (rmin + rmax) // 2

    new_rmin = max(0, center_y - new_height // 2)
    new_rmax = min(mask_array.shape[0], center_y + new_height // 2)
    new_cmin = max(0, center_x - new_width // 2)
    new_cmax = min(mask_array.shape[1], center_x + new_width // 2)

    new_mask_array = np.zeros_like(mask_array)
    new_mask_array[new_rmin:new_rmax, new_cmin:new_cmax] = 255

    new_mask_image = Image.fromarray(new_mask_array)
    return new_mask_image

def generate_min_mask(transformed_mask, object_image):
    mask_array = np.array(transformed_mask)

    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    object_width, object_height = object_image.size
    aspect_ratio = object_width / object_height

    original_width = cmax - cmin
    original_height = rmax - rmin

    if original_width / original_height < aspect_ratio:
        new_width = int(original_height * aspect_ratio)
        new_height = original_height
    else:
        new_height = int(original_width / aspect_ratio)
        new_width = original_width

    center_x = (cmin + cmax) // 2
    center_y = (rmin + rmax) // 2

    new_rmin = max(0, center_y - new_height // 2)
    new_rmax = min(mask_array.shape[0], center_y + new_height // 2)
    new_cmin = max(0, center_x - new_width // 2)
    new_cmax = min(mask_array.shape[1], center_x + new_width // 2)

    new_mask_array = np.zeros_like(mask_array)
    new_mask_array[new_rmin:new_rmax, new_cmin:new_cmax] = 255

    new_mask_image = Image.fromarray(new_mask_array)
    return new_mask_image

# Crop the excess parts
def crop_to_subject(image_path, mask):
    image = image_path.convert("RGBA")
    coords = np.column_stack(np.where(mask))

    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)

    cropped_image = image.crop((top_left[1], top_left[0], bottom_right[1], bottom_right[0]))
    return cropped_image

# Get the position of the mask
def fill_mask_area_white(background, mask):
    mask_np = np.array(mask)
    
    coords = np.column_stack(np.where(mask_np > 0))
    
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)
    
    draw = ImageDraw.Draw(background)
    
    return background, top_left, bottom_right

# Scale the subject image and paste it onto the background
def paste_object_to_white_area(background, object_img, top_left, bottom_right):
    target_width = bottom_right[1] - top_left[1]
    target_height = bottom_right[0] - top_left[0]

    obj_width, obj_height = object_img.size

    width_ratio = target_width / obj_width
    height_ratio = target_height / obj_height
    scale_ratio = min(width_ratio, height_ratio)

    new_width = int(obj_width * scale_ratio)
    new_height = int(obj_height * scale_ratio)
    object_resized = object_img.resize((new_width, new_height), Image.LANCZOS)

    paste_x = top_left[1] + (target_width - new_width) // 2
    paste_y = top_left[0] + (target_height - new_height) // 2

    background.paste(object_resized, (paste_x, paste_y), object_resized)
    
    return background, object_resized, (paste_x, paste_y)

# Update the mask (only keep the gap between the square and the subject)
def update_mask(mask, object_img, position):
    object_alpha = object_img.split()[-1]

    object_mask = Image.new("L", mask.size, 0)
    object_mask.paste(object_alpha, position)

    mask_np = np.array(mask)
    mask_np = np.where(mask_np >= 128, 255, 0)

    object_mask_np = np.array(object_mask)
    object_mask_np = np.where(object_mask_np >= 128, 255, 0)

    gap_mask_np = np.where((mask_np == 255) & (object_mask_np == 0), 255, 0).astype(np.uint8)
   
    gap_mask = Image.fromarray(gap_mask_np)
    return gap_mask

def extract_inverse_masked_image_opencv(image, mask):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    inverse_mask = cv2.bitwise_not(mask)
    masked_image = cv2.bitwise_and(image, image, mask=inverse_mask)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    masked_image = Image.fromarray(masked_image)
    return masked_image

def paste_irregular_subject(pasted_image, mask, result2):
    pasted_image_np = np.array(pasted_image)
    mask_np = np.array(mask)
    result2_np = np.array(result2)
    
    mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)
    
    y_nonzero, x_nonzero = np.nonzero(mask_np)
    
    for y, x in zip(y_nonzero, x_nonzero):
        result2_np[y, x] = pasted_image_np[y, x]
    
    result_image = Image.fromarray(result2_np)
    return result_image

remover = Remover(mode='base')

# --------- Step 5: Integration Process ---------
def full_image_fusion_pipeline(origin_image_path, ref_image_path, pipe, file_name, output_path):
    ref_image_foreground = remover.process(Image.open(ref_image_path), type='rgba')
    origin_image_foreground = remover.process(Image.open(origin_image_path), type='rgba')
    transformed_mask = resize_and_mask(ref_image_path, ref_image_foreground)
    subject_mask = get_alpha_mask(origin_image_foreground)
    cropped_object = crop_to_subject(origin_image_foreground, subject_mask)
    new_mask = generate_square_mask(transformed_mask, cropped_object)
    white_filled_background, top_left, bottom_right = fill_mask_area_white(Image.open(ref_image_path), new_mask.convert("L"))
    pasted_image, pasted_object, paste_position = paste_object_to_white_area(white_filled_background, cropped_object, top_left, bottom_right)
    updated_mask = update_mask(Image.fromarray(transformed_mask), pasted_object, paste_position)
    numpy_array = np.array(updated_mask)
    opencv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    updated_mask = cv2.dilate(opencv_image, kernel, iterations=2)
    updated_mask = Image.fromarray(updated_mask)
    pasted_object = pasted_object.split()[-1]
    pasted_image.resize((512, 512))
    updated_mask.resize((512, 512))
    pasted_image_array = np.array(pasted_image)
    updated_mask_array = np.array(updated_mask)
    pasted_image_array[updated_mask_array == 255] = 0
    init_image = Image.fromarray(pasted_image_array)
    prompt = 'neat, clean, tidy, briefness, food image'
    negative_prompt = "poor details, blurry, English, word"
    result = pipe(
        prompt=prompt, 
        image=init_image,
        mask_image=updated_mask,
        control_image=updated_mask,
        strength=1.0, 
        negative_prompt=negative_prompt,
        num_inference_steps=30,
    ).images[0].resize(white_filled_background.size)
    result.paste(pasted_object, paste_position, pasted_object)
    result.save(output_path + file_name + '.png')
