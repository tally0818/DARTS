from macro_structure import *
import torch.optim as optim

class DARTS():
  def __init__(self, op_types, N, input_size, output_size, BackBone, init_channels, train_loader, test_loader):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.op_types = op_types
    self.N = N
    self.input_size = input_size
    self.output_size = output_size
    self.BackBone = BackBone
    self.init_channels = init_channels
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.num_edges = int((self.N -1)*(self.N-2)/2)-1
    self.alpha = nn.Parameter(
            torch.zeros(2, self.num_edges, len(self.op_types), requires_grad=True).to(self.device)
        )
    self.discrete_ops = None

  def get_architecture(self):
    def create_normal_cell(in1_channels, in2_channels, previous_red_flag):
      return Cell(self.op_types, self.N, self.alpha[0], in1_channels, in2_channels, False, previous_red_flag)

    def create_reduction_cell(in1_channels, in2_channels, previous_red_flag):
      return Cell(self.op_types, self.N, self.alpha[1], in1_channels, in2_channels, True, previous_red_flag)

    architecture = Architecture(self.input_size,
                                self.output_size,
                                self.BackBone,
                                create_normal_cell,
                                create_reduction_cell,
                                self.init_channels)
    return architecture.to(self.device)

  def search_continuous_cells(self,
                              w_learning_rate = 0.025,
                              alpha_learning_rate = 0.001,
                              eta = 0.01,
                              epsilon = 0.01,
                              epochs = 5):
    model = self.get_architecture()
    w_optimizer = optim.SGD(model.parameters(), lr=w_learning_rate, momentum=0.9, weight_decay=3e-4)
    alpha_optimizer = optim.Adam([self.alpha], lr=alpha_learning_rate, betas=(0.5, 0.999), weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(w_optimizer, epochs)
    criterion = nn.CrossEntropyLoss().to(self.device)

    for epoch in range(epochs):
      epoch_train_loss = 0
      epoch_valid_loss = 0
      model.train()
      for batch_idx, ((train_inputs, train_targets), (valid_inputs, valid_targets)) in enumerate(zip(self.train_loader, self.test_loader)):
        train_inputs, train_targets = train_inputs.to(self.device), train_targets.to(self.device)
        valid_inputs, valid_targets = valid_inputs.to(self.device), valid_targets.to(self.device)
        # Phase1 : update architecture parameter
        alpha_optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_loss = criterion(train_outputs, train_targets)
        w_grad = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)

        old_w = [w.data.clone() for w in model.parameters()]

        with torch.no_grad():
          for w, grad in zip(model.parameters(), w_grad):
            w.data.add_(-eta * grad)

        valid_outputs = model(valid_inputs)
        valid_loss = criterion(valid_outputs, valid_targets)
        valid_alpha_grad = torch.autograd.grad(valid_loss, self.alpha, allow_unused=True)[0]

        with torch.no_grad():
          for p, w in zip(model.parameters(), old_w):
            p.data.copy_(w)
          for w, grad in zip(model.parameters(), w_grad):
            w.data.add_(epsilon * grad)

        train_outputs = model(train_inputs)
        train_loss_p = criterion(train_outputs, train_targets)
        alpha_grad_p = torch.autograd.grad(train_loss_p * eta / (2*epsilon), self.alpha, allow_unused=True)[0]

        with torch.no_grad():
          for p, w in zip(model.parameters(), old_w):
            p.data.copy_(w)
          for w, grad in zip(model.parameters(), w_grad):
            w.data.add_(-epsilon * grad)

        train_outputs = model(train_inputs)
        train_loss_m = criterion(train_outputs, train_targets)
        alpha_grad_m = torch.autograd.grad(train_loss_m * eta / (2*epsilon), self.alpha, allow_unused=True)[0]

        alpha_grad = (valid_alpha_grad - alpha_grad_p + alpha_grad_m)
        self.alpha.grad = alpha_grad
        alpha_optimizer.step()
        # Phase 2 : update architecture weights
        with torch.no_grad():
          for p, w in zip(model.parameters(), old_w):
            p.data.copy_(w)
        w_optimizer.zero_grad()
        model.update_alpha(self.alpha)
        train_outputs = model(train_inputs)
        train_loss = criterion(train_outputs, train_targets)
        train_loss.backward()
        w_optimizer.step()

        epoch_train_loss += train_loss.item()
        epoch_valid_loss += valid_loss.item()

        if batch_idx % 100 == 0:
          print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                f'Train Loss: {train_loss.item():.4f}, '
                f'Valid Loss: {valid_loss.item():.4f}')

      scheduler.step()
      epoch_train_loss /= len(self.train_loader)
      epoch_valid_loss /= len(self.test_loader)
      print(f'Epoch: {epoch}, '
            f'Train Loss: {epoch_train_loss:.4f}, '
            f'Valid Loss: {epoch_valid_loss:.4f}')
    model.alpha = self.alpha
    return model

  def _get_discrete_ops(self):
    discrete_ops = [[], []]
    for cell_type in range(2):
      tmp = [[] for _ in range(self.N-1)]
      edge_idx = 0
      for i in range(self.N-2):
        for j in range(i+1, self.N-1):
          if i == 0 and j == 1:
            continue
          if j > 1:
            top1 = torch.topk( torch.softmax(self.alpha[cell_type][edge_idx], dim=0)[:-1], 1)
            tmp[j].append([i, j, self.op_types[top1.indices.item()], top1.values.item()])
          edge_idx += 1

      for j in range(2, self.N-1):
        tmp_tmp = sorted(tmp[j], key=lambda x:x[-1], reverse = True)[:2]
        discrete_ops[cell_type].append(tmp_tmp[0][:-1])
        discrete_ops[cell_type].append(tmp_tmp[1][:-1])
    self.discrete_ops = discrete_ops
    return discrete_ops

  def get_discrete_model(self, model: Architecture):
    self._get_discrete_ops()
    assert self.discrete_ops != None
    for cell in model.cells:
      cell_type = cell.cell_type
      discrete_ops = self.discrete_ops[cell_type]
      for edge in cell.edges:
        kept_ops = [op for op in edge.ops if [edge.i, edge.j, op.op_type] in discrete_ops]
        edge.ops = kept_ops
      cell.edges = nn.ModuleList([edge for edge in cell.edges if len(edge.ops) > 0])
    return model