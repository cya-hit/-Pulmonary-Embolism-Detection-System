import sys
from pulmonary_embolism_final.utlis.ct_sample_sequence_converter import \
    reconstruct_semantic_from_sample_sequence
import Tool_Functions.Functions as Functions
# import format_convert.dcm_np_converter_new as dcm_to_np
import pulmonary_embolism_v2.transformer_PE_4D.predict_vessel_sequence as predict
import numpy as np
import pulmonary_embolism.prepare_dataset.get_branch_mask as get_branch_mask
from sementic_segmentation.predict_chest_semantics import predict_whole_av
import analysis.center_line_and_depth_3D as depth_and_center_line
import analysis.other_functions as other_functions
from pulmonary_embolism_final.performance.get_av_classification_metrics import process_to_get_metrics
from scipy.stats import gaussian_kde
import format_convert.spatial_normalize as spatial_normalize
import nibabel as nib
import gradio as gr
import matplotlib.pyplot as plt


# def histogram_list(value_list, interval=None, save_path=None, show=True, range_show=None,
#                    x_name='x', y_name='y', title='Histogram Plot', value_loc=1):
#     assert len(value_list) > 0
#     if interval is None:
#         interval = int(math.sqrt(len(value_list))) + 1
#
#     plt.hist(value_list, interval, range=range_show)
#     # plt.style.use('seaborn-poster')
#     plt.title(title)
#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#
#     height_arrow = int(len(value_list) / interval) * 3
#
#     plt.arrow(value_loc, height_arrow * 1.2, 0, -height_arrow, head_width=0.15, head_length=0.2 * height_arrow,
#               color='C1', alpha=1)
#
#     if save_path is not None:
#         plt.savefig(save_path)
#     if show:
#         plt.show()
#     plt.close()


def reformat_av(value_list):
    new_list = []
    for value in value_list:
        if 0 < value < np.inf:
            modified_value = (value + 1 / value) / 2
            new_list.append(modified_value)

    return new_list


def predict_embolism(nii_path):
    file_load = nib.load(nii_path)
    ct_data = np.array(file_load.get_fdata())
    # plt.imshow(ct_data[:, :, 150], cmap="gray")
    # plt.show()
    ct_data = np.transpose(ct_data, (1, 0, 2))
    ct_data = np.clip(ct_data, -1000, 2000)
    # plt.imshow(ct_data[:, :, 250], cmap="gray")
    # plt.show()

    rescaled_ct = (ct_data + 600) / 1600
    # plt.imshow(rescaled_ct[:, :, 250], cmap="gray")
    # plt.show()

    rescaled_ct = spatial_normalize.rescale_to_standard(rescaled_ct, [0.8, 0.8, 1], change_z_resolution=True)
    # plt.imshow(rescaled_ct[:, :, 250], cmap="gray")
    # plt.show()

    if rescaled_ct.shape[-1] % 2 == 1:
        rescaled_ct = rescaled_ct[:, :, :-1]

    # np.savez_compressed("/data/temp/non_contrast.npz", rescaled_ct)
    # exit()
    # artery, vein = seg_artery_vein_airway.predict_av_rescaled(rescaled_ct)
    # blood_vessel_mask = np.clip(artery + vein, 0, 1)

    # artery = np.load("/data/temp/non_contrast_a.npz")["arr_0"]
    # vein = np.load("/data/temp/non_contrast_v.npz")["arr_0"]

    ct_reclip = rescaled_ct * 1600 - 600
    ct_reclip = np.clip((ct_reclip + 1000) / 1600, 0, 1)

    # plt.imshow(ct_reclip[:, :, 250], cmap="gray")
    # plt.show()

    artery, vein, _ = predict_whole_av(ct_reclip)

    blood_vessel_mask = np.clip(artery + vein, 0, 1)
    blood_vessel_mask = np.array(blood_vessel_mask, 'float32')
    blood_vessel_mask = other_functions.smooth_mask(blood_vessel_mask)
    depth_array = depth_and_center_line.get_surface_distance(blood_vessel_mask, strict=True)
    blood_center_line = depth_and_center_line.get_center_line(blood_vessel_mask, surface_distance=depth_array)

    branch_map = get_branch_mask.get_branching_cloud(blood_center_line, depth_array, search_radius=5,
                                                     smooth_radius=1,
                                                     step=1, weight_half_decay=20, refine_radius=4)

    model_path = './Data_and_Models/vi_0.014_dice_0.608_precision_phase_model_guided.pth'

    model = predict.load_saved_model_guided(model_path=model_path)

    metrics, sample_sequence = process_to_get_metrics(
        rescaled_ct, depth_array, artery, vein, branch_map, blood_region_strict=None, model=model, return_sequence=True)
    # metrics, sample_sequence = process_to_get_metrics(
    #     rescaled_ct, depth_array, artery, vein, branch_map, blood_region_strict=None, model=model, return_sequence=True)

    metric_value = metrics['v0']

    distribution_list = Functions.pickle_load_object('./Data_and_Models/distribution_non_PE.pickle')
    # rank = get_rank_count(metric_value, distribution_list)

    distribution_list = [x for x in distribution_list if not (np.isinf(x) or np.isnan(x))]
    kde = gaussian_kde(np.array(distribution_list))
    p_value = (1 - kde.evaluate(metric_value)[0]) * 100
    p_value = round(p_value, 1)

    # print("higher a-v clot ratio means more possibility of pulmonary embolism")
    # print("in our data set of 4737 non-PE CT, this scan is higher than:", rank / 4737 * 100, '% of these scans')

    # metric_value = (metric_value + 1 / metric_value) / 2
    # histogram_list(reformat_av(distribution_list), value_loc=metric_value)

    predict_clot_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence, (4, 4, 5), key='clot_prob_mask')

    predict_clot_mask *= artery

    # mask_sums = np.sum(predict_clot_mask, axis=(0, 1))
    # max_index = np.argmax(mask_sums)
    #
    # sample_mask = np.transpose(predict_clot_mask[:, :, max_index])
    # sample_ct = np.transpose(ct_reclip[:, :, max_index])
    #
    # color = [254 / 255, 80 / 255, 20 / 255]
    #
    # merged_img = np.zeros([512 * 2 + 10, 512, 3])
    # for i in range(3):
    #     merged_img[:512, :, i] = sample_ct
    #
    # for i in range(3):
    #     merged_img[10 + 512:, :, i] = sample_ct * (1 - sample_mask)
    #     merged_img[10 + 512:, :, i] += sample_mask * color[i]
    #
    # merged_img = np.clip(merged_img, 0, 1)

    return (predict_clot_mask, p_value)


# print(predict_embolism("/data/temp/embolism/chest_embolism.nii.gz"))

# demo = gr.Interface(
#     # gr.Markdown("""<h1 align="center"> HorusEye for CT denoising.</h1>"""),
#     # gr.Markdown(
#     #     """<h3>
#     #     A temp GUI for HorusEye Feel free to upload a .dcm file with 512×512 size and see the denoised results.
#     #     </h3>"""),
#     predict_embolism,
#     inputs=[gr.File(type="filepath", label="Upload nii File")],
#     outputs=[gr.Image(label="Visualization of pulmonary embolism",
#                       height=240, width=240 / 512 * (512 * 2 + 10)),
#
#              gr.Text(label="Possibility of pulmonary embolism (%)")],
#
#     examples=[["./Data_and_Models/chest_embolism.nii.gz"]],
#     # title="Artery-vein segmentation for non-contrast CT",
#     # description="A temp GUI for HiPaS. Feel free to upload a .nii file with a width and height of 512. You can see the segmentation from the axial and coronal plane, and download the nii.gz file of segmentataion results.",
#     # article="We use the default direction for the input. If your own file has a different direction, the performance can degrade. Please check it before you upload your file."
# )
#
# demo.launch(share=True)
