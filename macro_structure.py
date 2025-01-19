from micro_structure import *

class Architecture(nn.Module):
  def __init__(self, input_size, output_size, BackBone, create_normal_cell, create_reduction_cell, init_channels):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.BackBone = BackBone
    self.create_normal_cell = create_normal_cell
    self.create_reduction_cell = create_reduction_cell
    self.init_channels = init_channels
    self.channels = [init_channels, init_channels]
    self.stem = nn.Sequential(
        nn.Conv2d(input_size, init_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(init_channels)
    )
    self.cells = nn.ModuleList()
    self._get_cells()
    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(self.channels[-1], output_size)
    )
    self.discrete_ops = None

  def _get_cells(self):
    previous_red_flag = False
    for cell_idx, num_cells in enumerate(self.BackBone):
      for _ in range(num_cells):
        if cell_idx % 2 == 0:
          cell = self.create_normal_cell(self.channels[-2], self.channels[-1], previous_red_flag)
          previous_red_flag = False
        else:
          cell = self.create_reduction_cell(self.channels[-2], self.channels[-1], previous_red_flag)
          previous_red_flag = True
        self.cells.append(cell)
        self.channels.append(cell.out_channels)

  def update_alpha(self, alpha):
    for cell in self.cells:
      cell.update_alpha(alpha[cell.cell_type])

  def forward(self, x):
      x = self.stem(x)
      prev_prev = prev = x
      for cell in self.cells:
        out = cell(prev_prev, prev)
        prev_prev, prev = prev, out
      out = self.classifier(out)
      return F.softmax(out, dim = 1)