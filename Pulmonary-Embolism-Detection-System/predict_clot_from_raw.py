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
from itertools import count

# 用于存储访问次数的文件
VISITOR_COUNT_FILE = 'visitor_count.txt'


def load_visitor_count():
    try:
        with open(VISITOR_COUNT_FILE, 'r') as f:
            content = f.read().strip()
            if content:
                return int(content)
            else:
                return 0
    except FileNotFoundError:
        return 0


def save_visitor_count(count):
    with open(VISITOR_COUNT_FILE, 'w') as f:
        f.write(str(count))


# 使用无限迭代器作为计数器
visitor_counter = count(start=load_visitor_count())


def get_visitor_count():
    return next(visitor_counter)


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
    ct_data = np.transpose(ct_data, (1, 0, 2))
    ct_data = np.clip(ct_data, -1000, 2000)
    rescaled_ct = (ct_data + 600) / 1600
    rescaled_ct = spatial_normalize.rescale_to_standard(rescaled_ct, [0.8, 0.8, 1], change_z_resolution=True)
    if rescaled_ct.shape[-1] % 2 == 1:
        rescaled_ct = rescaled_ct[:, :, :-1]
    ct_reclip = rescaled_ct * 1600 - 600
    ct_reclip = np.clip((ct_reclip + 1000) / 1600, 0, 1)
    artery, vein, _ = predict_whole_av(ct_reclip)
    blood_vessel_mask = np.clip(artery + vein, 0, 1)
    blood_vessel_mask = np.array(blood_vessel_mask, 'float32')
    blood_vessel_mask = other_functions.smooth_mask(blood_vessel_mask)
    depth_array = depth_and_center_line.get_surface_distance(blood_vessel_mask, strict=True)
    blood_center_line = depth_and_center_line.get_center_line(blood_vessel_mask, surface_distance=depth_array)
    branch_map = get_branch_mask.get_branching_cloud(blood_center_line, depth_array, search_radius=5,
                                                     smooth_radius=1,
                                                     step=1, weight_half_decay=20, refine_radius=4)
    model_path = '/mnt/data12t/Out_Share_PE/Data_and_Models/vi_0.014_dice_0.608_precision_phase_model_guided.pth'
    model = predict.load_saved_model_guided(model_path=model_path)
    metrics, sample_sequence = process_to_get_metrics(
        rescaled_ct, depth_array, artery, vein, branch_map, blood_region_strict=None, model=model, return_sequence=True)
    metric_value = metrics['v0']
    distribution_list = Functions.pickle_load_object('/mnt/data12t/Out_Share_PE/Data_and_Models/distribution_non_PE.pickle')
    distribution_list = [x for x in distribution_list if not (np.isinf(x) or np.isnan(x))]
    kde = gaussian_kde(np.array(distribution_list))
    p_value = (1 - kde.evaluate(metric_value)[0]) * 100
    p_value = round(p_value, 1)
    predict_clot_mask = reconstruct_semantic_from_sample_sequence(
        sample_sequence, (4, 4, 5), key='clot_prob_mask')
    predict_clot_mask *= artery
    mask_sums = np.sum(predict_clot_mask, axis=(0, 1))
    max_index = np.argmax(mask_sums)
    sample_mask = np.transpose(predict_clot_mask[:, :, max_index])
    sample_ct = np.transpose(ct_reclip[:, :, max_index])
    color = [254 / 255, 80 / 255, 20 / 255]
    merged_img = np.zeros([512 * 2 + 10, 512, 3])
    for i in range(3):
        merged_img[:512, :, i] = sample_ct
    for i in range(3):
        merged_img[10 + 512:, :, i] = sample_ct * (1 - sample_mask)
        merged_img[10 + 512:, :, i] += sample_mask * color[i]
    merged_img = np.clip(merged_img, 0, 1)
    return (np.transpose(merged_img, (1, 0, 2)),
            p_value,)


def predict_embolism_with_count(nii_path):
    current_count = get_visitor_count()
    result = predict_embolism(nii_path)
    save_visitor_count(current_count)  # 实时保存
    return result + (str(current_count),)


# 新增一个函数用于在页面加载时更新访问次数
def on_load():
    current_count = get_visitor_count()
    save_visitor_count(current_count)
    return str(current_count)


with gr.Blocks() as demo:

    # 使用 HTML 标签调整样式
    gr.Markdown("""
        <h1 style="text-align: center; color: #007BFF;">欢迎您使用肺栓塞检测系统</h1>
        <p style="text-align: center; font-size: 18px;">请提交.nii.gz格式文件</p>
        <p style="text-align: center; font-size: 18px;">上传文件后点击Submit开始预测，清除已上传文件点击Clear</p>
    """)
    # 添加分隔线
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            input_file = gr.File(type="filepath", label="Upload nii File")
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear")
            gr.Examples(
                examples=[["/mnt/data12t/Out_Share_PE/Data_and_Models/chest_embolism.nii.gz"]],
                inputs=input_file,
                outputs=[],
                fn=predict_embolism_with_count
            )
        with gr.Column():
            output_image = gr.Image(label="Visualization of pulmonary embolism",
                                    height=240, width=240 / 512 * (512 * 2 + 10))
            output_text = gr.Text(label="Possibility of pulmonary embolism (%)")

    output_count = gr.Text(label="Number of visitors", value=str(load_visitor_count()))

    submit_button.click(predict_embolism_with_count, inputs=input_file, outputs=[output_image, output_text, output_count])
    clear_button.click(lambda: (None, None, str(load_visitor_count())),
                       inputs=None, outputs=[output_image, output_text, output_count])

    demo.load(on_load, inputs=None, outputs=output_count)

demo.launch(share=True, inbrowser=True, server_name='192.168.1.105', server_port=7861)