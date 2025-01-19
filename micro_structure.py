from operations import *

class Edge(nn.Module):
  def __init__(self, i, j, operations, alpha):
    super().__init__()
    self.i = i
    self.j = j
    self.ops = operations
    self.alpha = alpha
    self.out_channels = self._calculate_out_channels()

  def _calculate_out_channels(self):
    return max([op.C_out for op in self.ops])

  def _get_outsize(self, x):
    b, c, h, w = x.size()
    stride = self.ops[0].stride
    return (b, self.out_channels, h // stride, w // stride)

  def _match_shapes(self, target_size, x):
    b, c_target, h_target, w_target = target_size
    _, c, h, w = x.size()
    device = x.device
    stride = 1
    if h != h_target:
      stride = h // h_target
    if c != c_target or stride != 1:
      conv = nn.Conv2d(c, c_target, 1, stride, 0).to(device)
      x = conv(x)
    return x

  def forward(self, x):
    out_size = self._get_outsize(x)
    out = torch.zeros(out_size, device=x.device)
    if len(self.ops) > 1:
      op_strengths = torch.softmax(self.alpha, dim=0)
      for op, strength in zip(self.ops, op_strengths):
        op_out = op(x)
        matched_out = self._match_shapes(out_size, op_out)
        out += matched_out * strength
      return out
    return self._match_shapes(out_size, self.ops[0](x))

class Cell(nn.Module):
  def __init__(self, op_types, N, alpha, in1_channels, in2_channels, reduction: bool, previous_red_flag : bool):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.op_types = op_types
    self.alpha = alpha
    self.N = N # 2 input nodes + 1 output node + N-3 intermediate nodes
    self.in1_channels = in1_channels
    self.in2_channels = in2_channels
    self.reduction = reduction
    self.previous_red_flag = previous_red_flag
    self.cell_type = int(self.reduction)  # 0 for normal, 1 for reduction
    self.reduction_factor = 2 if self.reduction else 1
    self.edges = nn.ModuleList()
    self.intermediate_channels = self._calculate_intermediate_channels()
    self._build_edges()
    self.out_channels = self._calculate_out_channels()
    self.to(self.device)

  def _calculate_intermediate_channels(self):
    return self.reduction_factor * max([self.in1_channels, self.in2_channels])

  def update_alpha(self, alpha):
    alpha_idx = 0
    for edge in self.edges:
      edge.alpha = self.alpha[alpha_idx]
      alpha_idx += 1

  def _get_input_channels(self, i):
    if self.previous_red_flag:
      return self.intermediate_channels
    if i == 0:
      return self.in1_channels
    if i == 1:
      return self.in2_channels
    return self.intermediate_channels

  def _build_edges(self):
    alpha_idx = 0
    for i in range(self.N-2):
      for j in range(i+1,self.N-1):
        if i == 0 and j == 1:
          continue
        if i > 1:
          self.reduction = False
          self.reduction_factor = 1
        operations = []
        for op_type in self.op_types:
          op = Operation(self._get_input_channels(i),
                         self.intermediate_channels,
                         op_type)
          op.stride = self.reduction_factor
          operations.append(op)
        edge = Edge(i, j, operations, self.alpha[alpha_idx])
        self.edges.append(edge)
        alpha_idx += 1

  def _calculate_out_channels(self):
    return (self.N - 3) * self.intermediate_channels

  def _match_shapes(self, target_size, x):
    b, c_target, h_target, w_target = target_size
    _, c, h, w = x.size()
    device = x.device
    stride = 1
    if h != h_target:
      stride = h // h_target
    if stride != 1:
      conv = nn.Conv2d(c, c_target, 1, stride, 0).to(device)
      x = conv(x)
    return x

  def forward(self, prev_prev, prev):
    states = [None] * (self.N-1)
    states[0] = prev_prev
    states[1] = prev
    states[0] = self._match_shapes(prev.size(), prev_prev)
    for edge in self.edges:
      if states[edge.j] is None:
        states[edge.j] = edge(states[edge.i])
      else:
        states[edge.j] += edge(states[edge.i])
    return torch.cat(states[2:], dim=1)