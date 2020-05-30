# Encoder
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Fire(nn.Module):
  """
    In channel: inplanes
    Out channel: expand1x1_planes + expand3x3_planes
  """

  def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
    super(Fire, self).__init__()
    self.inplanes = inplanes
    self.activation = nn.ReLU(inplace=True)
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.activation(self.squeeze(x))
    return torch.cat([self.activation(self.expand1x1(x)), self.activation(self.expand3x3(x))], 1)
    
# ******************************************************************************

class Backbone(nn.Module):
  """
     Class for Squeezeseg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params):
    # Call the super constructor
    super(Backbone, self).__init__()
    print("Using SqueezeNet Backbone")

    self.use_range = params["input_depth"]["range"]
    self.use_xyz = params["input_depth"]["xyz"]
    self.use_remission = params["input_depth"]["remission"]
    self.drop_prob = params["dropout"]
    self.OS = params["OS"]
    self.mobilenet = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    self.mobilenet.eval()

    # input depth calc
    self.input_depth = 0
    self.input_idxs = []
    if self.use_range:
      self.input_depth += 1
      self.input_idxs.append(0)
    if self.use_xyz:
      self.input_depth += 3
      self.input_idxs.extend([1, 2, 3])
    if self.use_remission:
      self.input_depth += 1
      self.input_idxs.append(4)
    # modified
    self.input_depth += 1 # proj_mask
    self.input_idxs.append(5)

    print("Depth of backbone input = ", self.input_depth)

    # stride play
    self.strides = [2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Original OS: ", current_os)

    # make the new stride
    if self.OS > current_os:
      print("Can't do OS, ", self.OS,
            " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      for i, stride in enumerate(reversed(self.strides), 0):
        if int(current_os) != self.OS:
          if stride == 2:
            current_os /= 2
            self.strides[-1 - i] = 1
        if int(current_os) == self.OS:
          break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)

    # encoder
    self.conv1a = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=3, stride=[1, self.strides[0]], padding=1), nn.ReLU(inplace=True))
    self.conv1b = nn.Conv2d(self.input_depth, 64, kernel_size=1, stride=1, padding=0)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=[1, self.strides[1]], padding=1)
    self.fire2 = Fire(64 + 32, 16, 64, 64) # fuse 7th
    self.fire3 = Fire(128, 16, 64, 64)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=[1, self.strides[2]], padding=1)
    self.fire4 = Fire(128 + 96, 32, 128, 128) # fuse 14th
    self.fire5 = Fire(256, 32, 128, 128)
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=[1, self.strides[3]], padding=1)
    self.fire6 = Fire(256, 48, 192, 192)
    self.fire7 = Fire(384 + 1280, 48, 192, 192) # fuse 19th
    self.fire8 = Fire(384, 64, 256, 256)
    self.fire9 = Fire(512, 64, 256, 256)

    # output
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 512

  def run_layer(self, x, layer, skips, os):
      y = layer(x)
      if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
        skips[os] = x.detach()
        os *= 2
      x = y
      return x, skips, os

  def run_mobilenet(self, x):
    rgb_features = {}
    for i, layer in enumerate(self.mobilenet.features):
      x = layer(x) # won't affect outer scope
      if i == 6:
        rgb_features['7th'] = x.clone().detach()
      elif i == 13:
        rgb_features['14th'] = x.clone().detach()
      elif i == 18:
        rgb_features['19th'] = x.clone().detach()
    return rgb_features

  def fill_missing_points(self, tensor, mask, kernelSize=3):
    """
    Fill missing points in `tensor` indicated by `mask`
    Args:
        tensor: any H * W tensor
        mask: boolean mask where True indicates keep original value
    Returns:
        combined: filled tensor
    """
    # TODO: deal with outliers using high pass filters: https://www.tutorialspoint.com/dip/high_pass_vs_low_pass_filters.htm
    eps = 1e-6
    k = kernelSize
    H, W = tensor.shape[0], tensor.shape[1]
    device = tensor.device
    tensor = tensor * mask # clear the tensor

    # apply median filter
    median = torch.zeros_like(tensor)
    for i in range(H):
      for j in range(W):
        window = tensor[max(0, i-k):min(H, i+k), max(0, j-k):min(W, j+k)]
        mask_w = mask[max(0, i-k):min(H, i+k), max(0, j-k):min(W, j+k)]
        if torch.sum(mask_w) > 0:
          median[i, j] = torch.median(window[mask_w])
    median = tensor + median * torch.logical_not(mask)

    # fill the top and bottom part
    # upperhalf: maximum
    still_missing_mask_top = torch.logical_and(median < eps, torch.cat((torch.full((H//2, W), True, dtype=torch.bool, device=device), torch.full((H//2, W), False, dtype=torch.bool, device=device))))
    maximum = torch.max(median.masked_select(torch.logical_not(still_missing_mask_top)))
    median.masked_fill_(still_missing_mask_top, maximum)
    # lowerhalf: minimum
    still_missing_mask_bottom = torch.logical_and(median < eps, torch.cat((torch.full((H//2, W), False, dtype=torch.bool, device=device), torch.full((H//2, W), True, dtype=torch.bool, device=device))))
    minimum = torch.min(median.masked_select(torch.logical_not(still_missing_mask_bottom)))
    median.masked_fill_(still_missing_mask_bottom, minimum)

    combined = tensor + median * torch.logical_not(mask)
    return combined
  
  def get_rgb_feature(self, range_img, rgb_features, calib_path):
    """get ready-to-fuse rgb features
    Note: Now only support batch size == 1!
    Args:
    range_img: batchsize=1 * ch * H * W tensor, channel = [range, x, y, z, remission, proj_mask]
    rgb_features: dict[rgb_layer_name] of rgb features, which are ch * H * W tensors
    Returns:
    corresponding_features: dict[sqseg_layer_name] of processed features
    """
    xyz = {}
    names = ['fire2', 'fire4', 'fire7']
    names_rgb = ['7th', '14th', '19th']
    strides = {'fire2':4, 'fire4':8, 'fire7':16}
    device = range_img.device
    assert range_img.shape[0] == 1, "batch size != 1, but now only support batch size == 1"
    range_img = range_img[0] # TODO: support arbitrary batch size
    a = range_img[1:4, :, ::2].clone().detach() # x, y, z
    mask = range_img[4, :, ::2].clone().detach().bool()
    for name in names:
      a, mask = a[:, :, ::2], mask[:, ::2] # only downsample on width
      li = []
      for i in range(3): # xyz
        li.append(self.fill_missing_points(a[i].clone(), mask)) # with missing points now
      xyz[name] = torch.stack(li, dim=0).reshape(3, -1) # flatten

    # read calib file
    calib = {}
    with open(calib_path, 'r') as file:
      for line in file.readlines():
        key, value = line.split(":", 1)
        value = torch.tensor([float(i) for i in value.split()], dtype=torch.float, device=device).reshape(3, 4)
        calib[key] = value
    Tr = torch.cat((calib['Tr'], torch.tensor([[0., 0., 0., 1.]], device=device)))
    P2 = calib['P2']

    # lidar_points -> RGB
    rgb_idx = {}
    for layer in names:
      # corresponding row and column on RGB
      rgb_idx[layer] = torch.mm(torch.mm(P2, Tr), torch.cat((xyz[layer], torch.ones((1, xyz[layer].shape[1]), device=device))))
      rgb_idx[layer][:2, :] /= rgb_idx[layer][2, :] # normalize
      rgb_idx[layer] = rgb_idx[layer][:2, :] # discard useless channel
      rgb_idx[layer] = rgb_idx[layer].reshape(2, range_img.shape[1], range_img.shape[2] // strides[layer]) # reshape to the same size as range feature
      # print(f"rgb_idx[{layer}].shape = {rgb_idx[layer].shape}")

    # clamp the out-of-bound points inside
    for na, nb in zip(names, names_rgb):
      H, W = rgb_features[nb].shape[2], rgb_features[nb].shape[3]
      # print(f"rgb_features[{nb}].shape = {rgb_features[nb].shape}")
      rgb_idx[na][0, :, :] = rgb_idx[na][0, :, :].clamp(min=0, max=H - 1)
      rgb_idx[na][1, :, :] = rgb_idx[na][1, :, :].clamp(min=0, max=W - 1)

    # retrieve RGB feature by correspondence (flow)
    # TODO: decide whether the rgb-features out-of-bound points get should be setted to 0
    flow = {}
    flow['fire2'] = torch.round(rgb_idx['fire2'] / 8).long() # mobilenetv2 7th
    flow['fire4'] = torch.round(rgb_idx['fire4'] / 16).long() # mobilenetv2 14th
    flow['fire7'] = torch.round(rgb_idx['fire7'] / 32).long() # mobilenetv2 19th

    corresponding_features = {}
    # print(f"7th: {rgb_features['7th'].shape}")
    # print(f"flow[fire2]: {flow['fire2'].shape}")
    corresponding_features['fire2'] = rgb_features['7th'][0, :, flow['fire2'][0, :], flow['fire2'][1, :]]
    # print(f"corresponding[fire2]: {corresponding_features['fire2'].shape}")
    corresponding_features['fire4'] = rgb_features['14th'][0, :, flow['fire4'][0, :], flow['fire4'][1, :]]
    corresponding_features['fire7'] = rgb_features['19th'][0, :, flow['fire7'][0, :], flow['fire7'][1, :]]

    return corresponding_features
    
  def forward(self, x, rgb_image, calib_path):
    # fuse preparation
    rgb_features = self.run_mobilenet(rgb_image)
    assert len(calib_path) == 1, "now only support batch size == 1"
    calib_path = calib_path[0] # only support batch size == 1
    features = self.get_rgb_feature(x, rgb_features, calib_path) # features ready to concat

    # filter input
    x = x[:, self.input_idxs]

    # run cnn
    # store for skip connections
    skips = {}
    os = 1

    # encoder
    skip_in = self.conv1b(x)
    x = self.conv1a(x)
    # first skip done manually
    skips[1] = skip_in.detach()
    os *= 2

    x, skips, os = self.run_layer(x, self.maxpool1, skips, os)
    # print(x.shape, features['fire2'].shape)
    x = torch.cat([x, features['fire2'].unsqueeze(0)], dim=1)
    x, skips, os = self.run_layer(x, self.fire2, skips, os)
    x, skips, os = self.run_layer(x, self.fire3, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.maxpool2, skips, os)
    x = torch.cat([x, features['fire4'].unsqueeze(0)], dim=1)
    x, skips, os = self.run_layer(x, self.fire4, skips, os)
    x, skips, os = self.run_layer(x, self.fire5, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.maxpool3, skips, os)
    x, skips, os = self.run_layer(x, self.fire6, skips, os)
    x = torch.cat([x, features['fire7'].unsqueeze(0)], dim=1)
    x, skips, os = self.run_layer(x, self.fire7, skips, os)
    x, skips, os = self.run_layer(x, self.fire8, skips, os)
    x, skips, os = self.run_layer(x, self.fire9, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)

    return x, skips
    
  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth