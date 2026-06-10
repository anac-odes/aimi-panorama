"""
prepare_dataset107.py

One-time preparation of Dataset107_PDAC_Detection for training the PDAC lesion
detector. Mirrors main.py steps 1-3 (Dataset103 pancreas localisation + pancreas-
ROI cropping) on the training data, and additionally crops the multi-class
PANORAMA labels with the same bboxes. Labels keep all 7 PANORAMA classes
(0=background, 1=PDAC_lesion, 2=veins, 3=arteries, 4=pancreas_parenchyma,
5=pancreatic_duct, 6=common_bile_duct) 

Prerequisites:
  - $nnUNet_raw/Dataset200_PANORAMA_full/imagesTr/     full CTs  (*_0000.nii.gz)
  - $nnUNet_raw/Dataset200_PANORAMA_full/labelsTr/     multi-class PANORAMA labels
  - $nnUNet_raw/Dataset200_PANORAMA_full/dataset.json  (will be copied verbatim)
  - $nnUNet_results/Dataset103_PANORAMA_baseline_Pancreas_Segmentation/...
        (pretrained pancreas-segmentation model from the released bundle)

Output:
  - $nnUNet_raw/Dataset107_PDAC_Detection/imagesTr/    cropped CTs  (*_0000.nii.gz)
  - $nnUNet_raw/Dataset107_PDAC_Detection/labelsTr/    cropped multi-class labels
  - $nnUNet_raw/Dataset107_PDAC_Detection/dataset.json (copy of source)
\
"""

import os
import os.path as osp
import json
import shutil
import subprocess
import warnings
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_file_extension(image_fp):
    base, ext = osp.splitext(image_fp)
    if ext == ".gz" and base.endswith(".nii"):
        return ".nii.gz"
    return ext


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False, out_size=[], out_origin=[], out_direction=[]):
    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()
    if not out_size:
        out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                    int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                    int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    if not out_direction:
        out_direction = itk_image.GetDirection()
    resample.SetOutputDirection(out_direction)
    if not out_origin:
        out_origin = itk_image.GetOrigin()
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    itk_image = resample.Execute(itk_image)
    return itk_image


def downsample_panorama_dataset(img_dir, img_save_dir, resample=(4.5, 4.5, 9.0)):
    assert osp.exists(img_dir), f'image directory does not exist: {img_dir}'
    if not osp.exists(img_save_dir):
        os.mkdir(img_save_dir)
    img_paths = sorted([f for f in glob(img_dir + '/*.*') if not f.endswith('.json')])
    if len(img_paths) == 0:
        print('No images found in input directory')
    with tqdm(total=len(img_paths)) as pbar:
        for img_path in img_paths:
            ext = get_file_extension(img_path)
            itk_img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            image_resampled = resample_img(itk_img, resample, is_label=False, out_size=[])
            sitk.WriteImage(image_resampled, osp.join(img_save_dir, osp.basename(img_path).replace(ext, '_0000.nii.gz')))
            pbar.update(1)


def crop_roi(img_dir, low_msk_dir, save_img_dir, margins=[100, 50, 15]):
    if not osp.exists(save_img_dir):
        os.mkdir(save_img_dir)
    img_paths = sorted(glob(img_dir + '/*.*'))
    crop_coordinates = {}
    with tqdm(total=len(img_paths)) as pbar:
        for img_path in img_paths:
            ext = get_file_extension(img_path)
            low_msk_path = osp.join(low_msk_dir, osp.basename(img_path).replace(ext, '.nii.gz'))
            img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            low_msk = sitk.ReadImage(low_msk_path)
            pancreas_mask_np = sitk.GetArrayFromImage(low_msk)
            pancreas_mask_np[pancreas_mask_np != 1] = 0
            pancreas_mask_np[pancreas_mask_np != 0] = 1
            pancreas_mask_nonzeros = np.nonzero(pancreas_mask_np)
            min_x = min(pancreas_mask_nonzeros[2])
            min_y = min(pancreas_mask_nonzeros[1])
            min_z = min(pancreas_mask_nonzeros[0])
            max_x = max(pancreas_mask_nonzeros[2])
            max_y = max(pancreas_mask_nonzeros[1])
            max_z = max(pancreas_mask_nonzeros[0])
            start_point_coordinates = (int(min_x), int(min_y), int(min_z))
            finish_point_coordinates = (int(max_x), int(max_y), int(max_z))
            start_point_physical = low_msk.TransformIndexToPhysicalPoint(start_point_coordinates)
            finish_point_physical = low_msk.TransformIndexToPhysicalPoint(finish_point_coordinates)
            start_point = img.TransformPhysicalPointToIndex(start_point_physical)
            finish_point = img.TransformPhysicalPointToIndex(finish_point_physical)
            spacing = img.GetSpacing()
            size = img.GetSize()
            marginx = int(margins[0] / spacing[0])
            marginy = int(margins[1] / spacing[1])
            marginz = int(margins[2] / spacing[2])
            x_start = max(0, start_point[0] - marginx)
            x_finish = min(size[0], finish_point[0] + marginx)
            y_start = max(0, start_point[1] - marginy)
            y_finish = min(size[1], finish_point[1] + marginy)
            z_start = max(0, start_point[2] - marginz)
            z_finish = min(size[2], finish_point[2] + marginz)
            cropped_image = img[x_start:x_finish, y_start:y_finish, z_start:z_finish]
            crop_coordinates[osp.basename(img_path).replace(ext, '')] = {
                'x_start': x_start,
                'x_finish': x_finish,
                'y_start': y_start,
                'y_finish': y_finish,
                'z_start': z_start,
                'z_finish': z_finish}
            sitk.WriteImage(cropped_image, osp.join(save_img_dir, osp.basename(img_path).replace(ext, '_0000.nii.gz')))
            pbar.update(1)
    return crop_coordinates


def predict(nnunet_model_dir, input_dir, output_dir, task: int, trainer: str = "nnUNetTrainer", plan: str = "nnUNetPlans",
            configuration="3d_fullres", checkpoint="checkpoint_final.pth",
            folds="0,1,2,3,4", store_probability_maps=True, tta=True):

    os.environ['RESULTS_FOLDER'] = str(nnunet_model_dir)
    cmd = [
        'nnUNetv2_predict',
        '-d',  str(task),
        '-i',  str(input_dir),
        '-o',  str(output_dir),
        '-c',  configuration,
        '-tr', trainer,
        '-p',  plan,
        '--continue_prediction'
    ]
    if folds:
        cmd.append('-f')
        cmd.extend(folds.split(','))
    if checkpoint:
        cmd.append('-chk')
        cmd.append(checkpoint)
    if store_probability_maps:
        cmd.append('--save_probabilities')
    if not tta:
        cmd.append('--disable_tta')

    cmd_str = " ".join(cmd)
    subprocess.check_call(cmd_str, shell=True)




def crop_labels(label_dir, crop_coordinates, save_lbl_dir, lesion_label=1):
    """For each case bbox produced by crop_roi, slice the matching multi-class
    label volume with the same indices. All 7 PANORAMA classes are preserved.

    crop_coordinates keys come from crop_roi: they are basenames of the source
    image files with the extension stripped, e.g. '100000_00001_0000'. The
    matching label file does NOT carry the '_0000' channel suffix, so strip it
    before looking up the label.

    Returns (n_pos, n_neg, n_skip) where n_pos counts cases containing
    `lesion_label` (class 1) -- useful for sanity-checking the lesion-positive
    fraction after cropping.
    """
    if not osp.exists(save_lbl_dir):
        os.mkdir(save_lbl_dir)
    n_pos = 0
    n_neg = 0
    n_skip = 0
    with tqdm(total=len(crop_coordinates)) as pbar:
        for case_id_with_channel, coords in crop_coordinates.items():
            if case_id_with_channel.endswith('_0000'):
                case_id = case_id_with_channel[:-len('_0000')]
            else:
                case_id = case_id_with_channel
            lbl_path = osp.join(label_dir, case_id + '.nii.gz')
            if not osp.isfile(lbl_path):
                print(f"  [skip] {case_id}: no label file at {lbl_path}")
                n_skip += 1
                pbar.update(1)
                continue
            lbl = sitk.ReadImage(lbl_path)
            cropped = lbl[coords['x_start']:coords['x_finish'],
                          coords['y_start']:coords['y_finish'],
                          coords['z_start']:coords['z_finish']]
            cropped_np = sitk.GetArrayFromImage(cropped)
            if (cropped_np == lesion_label).any():
                n_pos += 1
            else:
                n_neg += 1
            sitk.WriteImage(cropped, osp.join(save_lbl_dir, case_id + '.nii.gz'))
            pbar.update(1)
    return n_pos, n_neg, n_skip



def main():
    nnUNet_raw     = os.environ["nnUNet_raw"]
    nnUNet_results = os.environ["nnUNet_results"]

    SRC_NAME = "Dataset200_PANORAMA_full"
    DST_NAME = "Dataset107_PDAC_Detection"

    src_dir  = osp.join(nnUNet_raw, SRC_NAME)
    dst_dir  = osp.join(nnUNet_raw, DST_NAME)
    work_dir = osp.join(nnUNet_raw, "_dataset107_build_tmp")

    if not osp.exists(src_dir):
        raise SystemExit(
            f"source dataset not found: {src_dir}\n"
            f"Rename Dataset107_PDAC_Detection -> {SRC_NAME} first:\n"
            f"  mv {nnUNet_raw}/Dataset107_PDAC_Detection {src_dir}"
        )
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: downsample full CTs to low-resolution input for pancreas model
    print("\nStep 1/4: downsample full CTs")
    low_image_folder = osp.join(work_dir, 'LowImagesTr')
    downsample_panorama_dataset(osp.join(src_dir, "imagesTr"), low_image_folder)

    # Step 2: run Dataset103 to predict pancreas masks
    print("\nStep 2/4: predict pancreas masks with Dataset103 (5-fold ensemble)")
    low_pred_folder = osp.join(work_dir, 'LowPred')
    predict(
        nnunet_model_dir=nnUNet_results,
        input_dir=low_image_folder,
        output_dir=low_pred_folder,
        task=103,
    )

    # Step 3: crop full CTs to the pancreas ROI using the predicted masks
    print("\nStep 3/4: crop full CTs to the pancreas ROI")
    img_dst = osp.join(dst_dir, "imagesTr")
    crop_coordinates = crop_roi(
        img_dir=osp.join(src_dir, "imagesTr"),
        low_msk_dir=low_pred_folder,
        save_img_dir=img_dst,
        margins=[100, 50, 15],
    )

    # crop_roi appends a '_0000' suffix to the *output* filename. The source
    # filenames already end in '_0000.nii.gz' (the nnU-Net channel marker), so
    # cropped images currently end in '_0000_0000.nii.gz'. Fix that down to a
    # single '_0000.nii.gz' so Dataset107 is nnU-Net conformant.
    for fp in glob(osp.join(img_dst, "*_0000_0000.nii.gz")):
        new_fp = fp.replace("_0000_0000.nii.gz", "_0000.nii.gz")
        os.rename(fp, new_fp)

    # Step 4: crop the matching multi-class labels (keep all 7 classes)
    print("\nStep 4/4: crop multi-class labels with the same bboxes")
    lbl_dst = osp.join(dst_dir, "labelsTr")
    n_pos, n_neg, n_skip = crop_labels(
        label_dir=osp.join(src_dir, "labelsTr"),
        crop_coordinates=crop_coordinates,
        save_lbl_dir=lbl_dst,
        lesion_label=1,
    )

    # Copy the source dataset.json
    src_json = osp.join(src_dir, "dataset.json")
    dst_json = osp.join(dst_dir, "dataset.json")
    if osp.isfile(src_json):
        shutil.copy(src_json, dst_json)
        print(f"copied dataset.json from {src_json}")
    else:
        print(f"WARNING: no dataset.json found at {src_json} -- you will need to "
              f"create one manually at {dst_json}")
    n_total = len(glob(osp.join(img_dst, "*_0000.nii.gz")))

    shutil.rmtree(work_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"done. wrote {n_total} cases to {dst_dir}")
    print(f"  lesion-positive : {n_pos}")
    print(f"  lesion-negative : {n_neg}")
    print(f"  skipped         : {n_skip}")
    print(f"dataset.json      : {osp.join(dst_dir, 'dataset.json')}")
    print(f"next step         : nnUNetv2_plan_and_preprocess -d 107 -c 3d_fullres --verify_dataset_integrity")


if __name__ == "__main__":
    main()
