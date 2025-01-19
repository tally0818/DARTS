import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding):
    super().__init__()
    self.dep_conv = nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in)
    self.point_conv = nn.Conv2d(C_in, C_out, 1, 1, 0)
    self.bn = nn.BatchNorm2d(C_out)

  def forward(self, x):
    x = F.relu(x)
    x = self.dep_conv(x)
    x = self.point_conv(x)
    x = self.bn(x)
    return x

class Conv(nn.Module):
  '''
  '''
  def __init__(self, C_in, C_out, kernel_size, stride, padding):
    super().__init__()
    self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding)
    self.bn = nn.BatchNorm2d(C_out)

  def forward(self, x):
    x = F.relu(x)
    x = self.conv(x)
    x = self.bn(x)
    return x

class DilatedConv(nn.Module):
  def __init__(self, C_in : int, C_out : int, kernel_size : int, stride : int, dilation : int =2):
    super().__init__()
    padding = (kernel_size - 1) * dilation //2
    self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding, dilation=dilation)
    self.bn = nn.BatchNorm2d(C_out)

  def forward(self, x):
    x = F.relu(x)
    x = self.conv(x)
    x = self.bn(x)
    return x

class Zero(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return torch.zeros_like(x)

class Identity(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x

class Operation(nn.Module):
  def __init__(self, C_in, C_out, op_type):
    super().__init__()
    self.op_type = op_type
    self.C_in = C_in
    self.C_out = C_out
    self._stride = 1
    self.build_op()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  @property
  def stride(self):
    return self._stride

  @stride.setter
  def stride(self, value):
    self._stride = value
    self.build_op()
    self.to(self.device)

  def build_op(self):
    if self.op_type == '1x3_3x1_conv':
      self.op = nn.Sequential(
                Conv(self.C_in, self.C_out, (1,3), self.stride, (0,1)),
                Conv(self.C_out, self.C_out, (3,1), 1, (1,0))
            )
    elif self.op_type == '1x7_7x1_conv':
      self.op = nn.Sequential(
                Conv(self.C_in, self.C_out, (1,7), self.stride, (0,3)),
                Conv(self.C_out, self.C_out, (7,1), 1, (3,0))
            )
    elif self.op_type == '3x3_avgpool':
      self.op = nn.Sequential(
                nn.AvgPool2d(3, stride=self.stride, padding=1)
            )
    elif self.op_type == '3x3_maxpool':
      self.op = nn.Sequential(
                nn.MaxPool2d(3, stride=self.stride, padding=1)
            )
    elif self.op_type == '5x5_maxpool':
      self.op = nn.Sequential(
                nn.MaxPool2d(5, stride=self.stride, padding=2)
            )
    elif self.op_type == '7x7_maxpool':
      self.op = nn.Sequential(
                nn.MaxPool2d(7, stride=self.stride, padding=3)
            )
    elif self.op_type == '1x1_conv':
      self.op = Conv(self.C_in, self.C_out, 1, self.stride, 0)
    elif self.op_type == '3x3_conv':
      self.op = Conv(self.C_in, self.C_out, 3, self.stride, 1)
    elif self.op_type == '3x3_sepconv':
      self.op = nn.Sequential(
                SepConv(self.C_in, self.C_out, 3, self.stride, 1),
                SepConv(self.C_out, self.C_out, 3, 1, 1)
            )
    elif self.op_type == '5x5_sepconv':
      self.op = nn.Sequential(
                SepConv(self.C_in, self.C_out, 5, self.stride, 2),
                SepConv(self.C_out, self.C_out, 5, 1, 2)
            )
    elif self.op_type == '7x7_sepconv':
      self.op = nn.Sequential(
                SepConv(self.C_in, self.C_out, 7, self.stride, 3),
                SepConv(self.C_out, self.C_out, 7, 1, 3)
            )
    elif self.op_type == '3x3_dilconv':
      self.op = DilatedConv(self.C_in, self.C_out, 3, self.stride)
    elif self.op_type == '5x5_dilconv':
      self.op = DilatedConv(self.C_in, self.C_out, 5, self.stride)
    elif self.op_type == 'identity':
      self.op = Identity()
    else:
      self.op = Zero()

  def forward(self, x):
    return self.op(x)