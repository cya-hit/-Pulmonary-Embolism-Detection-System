import os
# from PIL import Image
import gradio as gr
from gradio_client import Client

client = Client("https://ff13f697993caffdf7.gradio.live")


def predict(nii_path):

    result = client.predict(nii_path)
    return result[0], result[1]


demo = gr.Interface(
    # gr.Markdown("""<h1 align="center"> HorusEye for CT denoising.</h1>"""),
    # gr.Markdown(
    #     """<h3>
    #     A temp GUI for HorusEye Feel free to upload a .dcm file with 512×512 size and see the denoised results.
    #     </h3>"""),
    predict,
    inputs=[gr.File(type="filepath", label="Upload nii File")],
    outputs=[gr.Image(label="Visualization of pulmonary embolism",
                      height=240, width=240 / 512 * (512 * 2 + 10)),

             gr.Text(label="Possibility of pulmonary embolism (%)")],

    examples=[["./chest_embolism.nii.gz"]],
    title="Predict embolism from non-contrast CT",
    # description="A temp GUI for HiPaS. Feel free to upload a .nii file with a width and height of 512. You can see the segmentation from the axial and coronal plane, and download the nii.gz file of segmentataion results.",
    # article="We use the default direction for the input. If your own file has a different direction, the performance can degrade. Please check it before you upload your file."
)

demo.launch(share=True)