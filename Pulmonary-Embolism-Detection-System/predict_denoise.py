import torch
import torch.nn as nn
import numpy as np
import pydicom
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
import SimpleITK as sitk
from monai.inferers import SliceInferer

# import sys
# print(sys.path)

class Getgradientnopadding(nn.Module):
    def __init__(self):
        super(Getgradientnopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x

class RED(nn.Module):
    def __init__(self, out_ch=64):
        super(RED, self).__init__()
        self.conv1 = nn.Conv2d(1 + 1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch * 2, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch * 4, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch * 2, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.get_g_nopadding = Getgradientnopadding()
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        grad = self.get_g_nopadding(x)

        residual_1 = x
        out = self.relu(self.conv1(torch.concatenate((x, grad), dim=1)))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out = residual_1 + out
        return out


class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out


class SwinUNet(nn.Module):
    def __init__(self, feature_ch=96, final_ch=1):
        super(SwinUNet, self).__init__()
        self.swin = SwinUNETR(img_size=(512, 512),
                              in_channels=2,
                              out_channels=final_ch,
                              depths=[2, 2, 18, 2],
                              spatial_dims=2,
                              use_checkpoint=True,
                              feature_size=feature_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(final_ch, 1, kernel_size=3, stride=1, padding=1),
        )
        self.get_g_nopadding = Getgradientnopadding()

    def forward(self, x):

        grad = self.get_g_nopadding(x)
        residual = x
        x = torch.concatenate((x, grad), dim=1)
        F = self.swin(x)
        out = self.conv(F)
        return out + residual


def predict_denoised_slice(ct_slice, model="HorusEye"):
    assert model in ["HorusEye", "REN_CNN", "CT_former"]

    if model == "HorusEye":
        denoise_model = SwinUNet()
        denoise_model = denoise_model.cuda()
        denoise_model.half()
        denoise_model.eval()
        slice = torch.tensor(ct_slice[np.newaxis, np.newaxis, :]).to(torch.float).to('cuda').half()
        denoise_model.load_state_dict(torch.load("/data/Model/denoise_V10/Swin_v3.pth"))
        # model.load_state_dict(torch.load(
        #     "/data/Model/denoise_V9/RED_200_grad.pth"))
        denoised = denoise_model(slice).cpu().detach().numpy()[0, 0]
        return np.array(denoised, "float32")
    elif model == "RED_CNN":
        # ct_scan = ct_slice * 1600 - 1000
        ct_slice = np.clip((ct_slice * 1600) / (3000 + 1000), 0, 1)

        ct_slice = torch.tensor(ct_slice[np.newaxis, np.newaxis, :]).to(torch.float).to("cuda").half()
        denoise_model = RED_CNN()
        denoise_model.load_state_dict(torch.load("/data/Model/denoise_V9/RED_nc.pth"))
        denoise_model = denoise_model.cuda()
        denoise_model.half()
        denoise_model.eval()
        prediction = denoise_model(ct_slice).cpu().detach().numpy()[0, 0]
        return prediction * 4000 / 1600
    else:
        # ct_slice = ct_slice * (high-low) + low
        ct_slice = np.clip(ct_slice * 1600 / (3000 + 1000), 0, 1)
        ct_scan = torch.tensor(ct_slice[np.newaxis, np.newaxis, :]).to(torch.float).to("cuda")
        denoise_model = CTformer(img_size=64, tokens_type='performer', embed_dim=64, depth=1, num_heads=8, kernel=4,
                                 stride=4, mlp_ratio=2., token_dim=64)
        denoise_model.load_state_dict(torch.load("/home/chuy/Downloads/T2T_vit_530000iter.ckpt"))
        denoise_model = denoise_model.cuda()

        prediction = np.zeros([512, 512])
        for i in range(0, 8):
            for j in range(0, 8):
                sub_scan = ct_scan[:, :, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]
                prediction[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] \
                    = denoise_model(sub_scan).cpu().detach().numpy()[0, 0]

        return prediction * 4000 / 1600


def predict_dicom(dicom_file, model="HorusEye", high=400, low=-1000, return_rescaled=False):
    dcm = pydicom.read_file(dicom_file)
    ct_data = np.array(dcm.pixel_array, "float32") - 1000
    ct_data = np.clip((ct_data - low) / (high - low), 0, 1)
    de = np.clip(predict_denoised_slice(ct_data, model), 0, 1)
    if return_rescaled:
        return np.array(de * (high - low) + low, "int32")
    else:
        return de


def predict_denoised_scan(ct_array, batch_size=8):
    model = SwinUNet().cuda()
    model.half()
    model.eval()

    input_set = torch.from_numpy(ct_array[np.newaxis, np.newaxis]).to(torch.float)
    input_set = input_set.to('cuda').half()
    model.load_state_dict(torch.load("/data/Model/denoise_V10/Swin_v2.pth"))
    with torch.no_grad():
        inferer = SliceInferer(spatial_dim=2,
                               roi_size=(512, 512),
                               sw_batch_size=batch_size,
                               progress=False)
        denoised = inferer(inputs=input_set, network=model).detach().cpu().numpy()[0, 0]
    denoised = np.array(denoised, "float32")

    for i in range(ct_array.shape[-1]):
        if np.sum(ct_array[:, :, i]) == 0:
            denoised[:, :, i] = 0

    return denoised


def predict_dicom_file(dicom_dict, high=400, low=-1000, return_rescaled=False):
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(dicom_dict)
    reader.SetFileNames(dcm_series)

    img = reader.Execute()
    ct_data = sitk.GetArrayFromImage(img)  # z y x
    ct_data = np.swapaxes(ct_data, 0, 2)
    ct_data = np.swapaxes(ct_data, 0, 1)

    ct_data = np.clip((ct_data - low) / (high - low), 0, 1)
    de = np.clip(predict_denoised_scan(ct_data), 0, 1)
    if return_rescaled:
        return np.array(de * (high - low) + low, "int32")
    else:
        return de


if __name__ == '__main__':
    predict_dicom('匿名.Seq4.Ser203.Img94.dcm')