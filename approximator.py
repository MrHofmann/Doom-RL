from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import math

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    #DEVICE = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')


class CustomLoss(torch.nn.Module):    
    def __init__(self):
        super(CustomLoss,self).__init__()
    
    def forward(self, predicted, target):
        abs_err = abs(predicted - target)
        #quadratic_part = tensor.minimum(abs_err, 1)
        quadratic_part = torch.min(abs_err, torch.tensor(1.0).to(DEVICE))
        linear_part = abs_err - quadratic_part
        loss = (0.5 * quadratic_part ** 2 + linear_part)
        return loss

class KLDivergence(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, target_dist, predicted_dist):
            dist_error = (target_dist * (target_dist / predicted_dist).log()).sum(1)

            return dist_error

#class ISLoss(torch.nn.Module):
#    def __init__(self):
#        super(ISLoss,self).__init__()
#    
#    def forward(self, predicted, target, ISWeights):
#        #print("loss")
#        #print(predicted)
#        #print(target)
#        abs_err = abs(predicted - target)
#        #print(abs_err)
#        #quadratic_part = tensor.minimum(abs_err, 1)
#        quadratic_part = torch.min(abs_err, torch.tensor(1.0).to(DEVICE))
#        #print(quadratic_part)
#        linear_part = abs_err - quadratic_part
#        #print(linear_part)
#       
#        #print("ISWeights")
#        #print((0.5 * quadratic_part ** 2 + linear_part).shape)
#        #print(ISWeights.reshape((64)))
#        #print(torch.from_numpy(ISWeights).shape)
#        #loss = torch.from_numpy(ISWeights)*(0.5 * quadratic_part ** 2 + linear_part)
#        #loss = torch.from_numpy(ISWeights.reshape((64)))*(0.5 * quadratic_part ** 2 + linear_part)
#        loss = ISWeights*(0.5 * quadratic_part ** 2 + linear_part)
#        #print(loss)
#        #print(loss2)
#        #return loss.sum()
#        return loss


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.1)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.1)
    elif isinstance(m, NoisyLinear):
        m.reset_parameters()
        m.reset_noise()


# GET TD UPDATE
def deepmind_rmsprop(loss_or_grads, params, learning_rate=0.00025,
                     rho=0.95, epsilon=0.01):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)

        acc_grad = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        acc_grad_new = rho * acc_grad + (1 - rho) * grad

        acc_rms = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                broadcastable=param.broadcastable)
        acc_rms_new = rho * acc_rms + (1 - rho) * grad ** 2

        updates[acc_grad] = acc_grad_new
        updates[acc_rms] = acc_rms_new

        updates[param] = (param - learning_rate *
                          (grad /
                           T.sqrt(acc_rms_new - acc_grad_new ** 2 + epsilon)))

    return updates

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features    = in_features
        self.out_features   = out_features
        self.std_init       = std_init

        self.weight_mu      = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma   = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))

        self.bias_mu        = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma     = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            # Had to edit this because it gives error when using Variable
            #weight  = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            #bias    = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
            weight  = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias    = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight  = self.weight_mu
            bias    = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1/math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init/math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init/math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class ActionCategoricalNetwork(nn.Module):
    def __init__(self, img_input_shape, output_size, ddqn=False, dueling=False, noisy=False):
        super(ActionCategoricalNetwork, self).__init__()

        self.num_actions = output_size[0]
        self.num_atoms = output_size[1]
        self.dueling = dueling
        self.noisy = noisy

        # Input size with batch is [64, 4, 84, 84].
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=True),
            #nn.BatchNorm2d(32),
            nn.ReLU()
            )

        # After conv1 [64, 32, 20, 20]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU()
            )
                
        # After conv2 [64, 64, 9, 9]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU()
            )        
        
        # Ater conv3 [64, 64, 7, 7]
        if dueling and noisy:
            self.advantage_fc = nn.Sequential(
                    NoisyLinear(3172, 256),
                    nn.ReLU(),
                    NoisyLinear(256, self.num_actions*self.num_atoms)
                    )
            self.state_fc = nn.Sequential(
                    NoisyLinear(3172, 256),
                    nn.ReLU(),
                    NoisyLinear(256, self.num_atoms)
                    )
        elif dueling:
            self.advantage_fc = nn.Sequential(
                nn.Linear(3172, 256, bias=True),
                #nn.Linear(1586, 256, bias=True),
                nn.ReLU(),
                nn.Linear(256, self.num_actions*self.num_atoms, bias=True)
                )
            self.state_fc = nn.Sequential(
                nn.Linear(3172, 256, bias=True),
                #nn.Linear(1586, 256, bias=True),
                nn.ReLU(),
                nn.Linear(256, self.num_atoms, bias=True)
                )
        elif noisy:
            self.fc1 = nn.Sequential(
                NoisyLinear(3172, 512),
                nn.ReLU()
                )
            self.out = nn.Sequential(NoisyLinear(512, self.num_actions * self.num_atoms))
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(3172, 512, bias=True),
                nn.ReLU()
                )
            self.out = nn.Sequential(nn.Linear(512, self.num_actions * self.num_atoms, bias=True))

    def reset_noise(self):
        if self.dueling:
            for m in self.advantage_fc.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
            for m in self.state_fc.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
        else:
            for m in self.fc1.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
            for m in self.out.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()

    def forward(self, x, misc=None):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.dueling:
            x = torch.cat((x.view(x.size(0), -1), misc), dim=1)
            state_dist = self.state_fc(x)#.reshape(-1, 1, self.num_atoms)
            advantage_dist = self.advantage_fc(x)#.reshape(-1, self.num_actions, self.num_atoms)
             
            state_dist = state_dist.view(batch_size, 1, self.num_atoms)
            advantage_dist = advantage_dist.view(batch_size, self.num_actions, self.num_atoms)

            x = state_dist + advantage_dist - advantage_dist.mean(1, keepdim=True)
        else:
            ##x = x.view(x.size()[0], -1)
            x = torch.cat((x.view(-1, 3136), misc), dim=1)
            x = self.fc1(x)
            x = self.out(x)

        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        return x


#class DuelingActionValueNetwork(nn.Module):
#    def __init__(self, img_input_shape, output_size, ddqn=False, noisy=False):
#        super(DuelingActionValueNetwork, self).__init__()
#        
#        # Input size with batch is [64, 4, 84, 84].
#        self.conv1 = nn.Sequential(
#            nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=True),
#            ##nn.BatchNorm2d(32),
#            nn.ReLU()
#            )
#
#        # After conv1 [64, 32, 20, 20]
#        self.conv2 = nn.Sequential(
#            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True),
#            #nn.BatchNorm2d(64),
#            nn.ReLU()
#            )
#                
#        # After conv2 [64, 64, 9, 9]
#        self.conv3 = nn.Sequential(
#            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True),
#            #nn.BatchNorm2d(64),
#            nn.ReLU()
#            )        
#        
#        if noisy:
#            self.advantage_fc = nn.Sequential(
#                NoisyLinear(3172, 256),
#                nn.ReLU(),
#                NoisyLinear(256, output_size)
#                )
#
#            self.state_fc = nn.Sequential(
#                NoisyLinear(3172, 256),
#                nn.ReLU(),
#                NoisyLinear(256, 1)
#                )
#        else:
#            self.advantage_fc = nn.Sequential(
#                nn.Linear(3172, 256, bias=True),
#                #nn.Linear(1586, 256, bias=True),
#                nn.ReLU(),
#                nn.Linear(256, output_size, bias=True)
#                )
#
#            self.state_fc = nn.Sequential(
#                nn.Linear(3172, 256, bias=True),
#                #nn.Linear(1586, 256, bias=True),
#                nn.ReLU(),
#                nn.Linear(256, 1, bias=True)
#                )
#
#            #self.out = nn.Sequential(nn.Linear(512, output_size, bias=True))
#    
#    #def reset_noise(self):
#    #    self.noisy1.reset_noise()
#    #    self.noisy2.reset_noise()
#
#    def forward(self, x, misc=None): 
#        x = self.conv1(x)
#        x = self.conv2(x)
#        x = self.conv3(x)
#        #x = torch.cat((x.view(-1, 3136), misc), dim=1)
#        ##x = self.fc1(x)
#        ##x = self.out(x)
#        #x1 = x[:, :1586]
#        #x2 = x[:, 1586:]
#        #state_value = self.state_fc(x1).reshape(-1, 1)
#        #advantage_values = self.advantage_fc(x2)
#        #x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))
#
#        x = torch.cat((x.view(x.size(0), -1), misc), dim=1)
#        state_value = self.state_fc(x)
#        advantage_values = self.advantage_fc(x)
#        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))
#
#        #print(x)
#        return x

class ActionValueNetwork(nn.Module):
    def __init__(self, img_input_shape, output_size, ddqn=False, dueling=False, noisy=False):
        super(ActionValueNetwork, self).__init__()

        self.dueling = dueling
        self.noisy = noisy

        # Input size with batch is [64, 4, 84, 84].
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=True),
            #nn.BatchNorm2d(32),
            nn.ReLU()
            )

        # After conv1 [64, 32, 20, 20]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU()
            )
                
        # After conv2 [64, 64, 9, 9]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU()
            )        
        
        # Ater conv3 [64, 64, 7, 7]
        if dueling and noisy:
            self.advantage_fc = nn.Sequential(
                NoisyLinear(3172, 256),
                nn.ReLU(),
                NoisyLinear(256, output_size)
                )
            self.state_fc = nn.Sequential(
                NoisyLinear(3172, 256),
                nn.ReLU(),
                NoisyLinear(256, 1)
                )
        elif dueling:                        
            self.advantage_fc = nn.Sequential(
                nn.Linear(3172, 256, bias=True),
                #nn.Linear(1586, 256, bias=True),
                nn.ReLU(),
                nn.Linear(256, output_size, bias=True)
                )
            self.state_fc = nn.Sequential(
                nn.Linear(3172, 256, bias=True),
                #nn.Linear(1586, 256, bias=True),
                nn.ReLU(),
                nn.Linear(256, 1, bias=True)
                )
        elif noisy:
            self.fc1 = nn.Sequential(
                NoisyLinear(3172, 512),
                nn.ReLU()
            )
            self.out = nn.Sequential(NoisyLinear(512, output_size))
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(3172, 512, bias=True),
                nn.ReLU()
            )
            self.out = nn.Sequential(nn.Linear(512, output_size, bias=True))

    def reset_noise(self):
        if self.dueling:
            for m in self.advantage_fc.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
            for m in self.state_fc.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
        else:
            for m in self.fc1.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
            for m in self.out.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()

    def forward(self, x, misc=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.dueling:
            #x = torch.cat((x.view(-1, 3136), misc), dim=1)
            ##x = self.fc1(x)
            ##x = self.out(x)
            #x1 = x[:, :1586]
            #x2 = x[:, 1586:]
            #state_value = self.state_fc(x1).reshape(-1, 1)
            #advantage_values = self.advantage_fc(x2)
            #x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

            #x = self.features(x)
            #x = x.view(x.size(0), -1)
            #advantage = self.advantage(x)
            #value     = self.value(x)
            #return value + advantage  - advantage.mean()

            x = torch.cat((x.view(x.size(0), -1), misc), dim=1)
            state_value = self.state_fc(x)
            advantage_values = self.advantage_fc(x)
            x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))
        else:
            #x = x.view(x.size()[0], -1)
            x = torch.cat((x.view(-1, 3136), misc), dim=1)
            x = self.fc1(x)
            x = self.out(x)

        return x


class DQN:
    def __init__(self, state_format, actions_number, gamma=0.99, learning_rate=0.00025, ddqn=False, **kwargs):
        self.inputs = dict()
        self.learning_rate = learning_rate
        architecture = kwargs

        self.loss_history = []
        self.misc_state_included = (state_format["s_misc"] > 0)
        self.gamma = np.float64(gamma)
        
        #https://pytorch.org/docs/stable/tensors.html
        self.inputs["S0"] = torch.tensor([])
        self.inputs["S1"] = torch.tensor([])
        self.inputs["A"] = torch.tensor([])
        self.inputs["R"] = torch.tensor([])
        self.inputs["Nonterminal"] = torch.tensor([])
        if self.misc_state_included:
            self.inputs["S0_misc"] = torch.tensor([])
            self.inputs["S1_misc"] = torch.tensor([])
            self.misc_len = state_format["s_misc"]
        else:
            self.misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self.single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self.misc_len
        architecture["output_size"] = actions_number

        if self.misc_state_included:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"],
                                                                     misc_input=self.inputs["S0_misc"],
                                                                     **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                misc_input=self.inputs["S1_misc"],
                                                                                **architecture)
        else:

            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"], **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                **architecture)

        self.alternate_input_mappings = {}
        for layer, input in zip(input_layers, alternate_inputs):
            self.alternate_input_mappings[layer] = input

        self.network.apply(init_weights)
        self.frozen_network.apply(init_weights)
        self.criterion = CustomLoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01)

# INITIALIZE
    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]

        if self.misc_state_included:
            inputs.append(misc_input)
        
        network = ActionValueNetwork(img_input_shape, output_size, dueling=False, noisy=False).to(DEVICE)
        return network, input_layers, inputs

# GET ACTION VALUES
    def estimate_best_action(self, state):
        if self.misc_state_included:
            state_img = torch.from_numpy(state[0].reshape(self.single_image_input_shape)).float().to(DEVICE)
            state_misc = torch.from_numpy(state[1].reshape(1, self.misc_len)).float().to(DEVICE)
            qvals = self.network(state_img,  state_misc)

            a = torch.argmax(qvals).item()
        else:
            state_img = torch.from_numpy(state[0].reshape(self.single_image_input_shape)).float().to(DEVICE)
            qvals = self.network(state_img)

            a = torch.argmax(qvals).item()

        return a

# OPTIMIZE
    def learn(self, transitions):
        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]
               
        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)
                #q2 = self.network(s2_img, s2_misc)
                #q2_action_ref = torch.argmax(q2, dim=1)
            
                #q2_frozen = self.frozen_network(s2_img, s2_misc)
                #q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]
            
            q2_max = torch.max(self.frozen_network(s2_img, s2_misc), dim=1)
            target_q = (r + self.gamma * nonterminal * q2_max[0]).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)
            
            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            td_error = td_error.sum()
            td_error.backward()
            self.optimizer.step()
        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(td_error.item())

# INITIALIZE
    @staticmethod
    def build_loss_expression(predicted, target):

        abs_err = abs(predicted - target)
        quadratic_part = torch.minimum(abs_err, torch.tensor(1))
        linear_part = abs_err - quadratic_part
        loss = (0.5 * quadratic_part ** 2 + linear_part)
        return loss

    def get_mean_loss(self, clear=True):
        m = np.mean(self.loss_history)
        if clear:
            self.loss_history = []
        return m

    def get_network(self):
        return self.network

    def melt(self):
        self.frozen_network.load_state_dict(self.network.state_dict())


class DoubleDQN(DQN):
# OPTIMIZE
    def learn(self, transitions):

        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]

        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)
            #with torch.no_grad():
            q2 = self.network(s2_img, s2_misc)
            q2_action_ref = torch.argmax(q2, dim=1)
            
            q2_frozen = self.frozen_network(s2_img, s2_misc)
            q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]

            target_q = (r + self.gamma * nonterminal * q2_max).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)

            #q_values      = current_model(state)
            #next_q_values = current_model(next_state)
            #next_q_state_values = target_model(next_state) 
            #
            #q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
            #next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            #expected_q_value = reward + gamma * next_q_value * (1 - done)
            # 
            #loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            td_error = td_error.sum()
            td_error.backward()
            self.optimizer.step()
        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(td_error.item())


class DuelingDQN(DQN):
    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]

        if self.misc_state_included:
            inputs.append(misc_input)
        
        network = ActionValueNetwork(img_input_shape, output_size, dueling=True, noisy=False).to(DEVICE)
        return network, input_layers, inputs


class PrioritizedDQN(DQN):
    def learn(self, transitions, ISWeights):       
        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]

        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)
            is_weights = torch.from_numpy(ISWeights).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)

                #q2 = self.network(s2_img, s2_misc)
                #q2_action_ref = torch.argmax(q2, dim=1)
            
                #q2_frozen = self.frozen_network(s2_img, s2_misc)
                #q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]
            
            q2_max = torch.max(self.frozen_network(s2_img, s2_misc), dim=1)
            target_q = (r + self.gamma * nonterminal * q2_max[0]).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)
         
            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            #td_error = self.criterion(predicted_q, target_q, is_weights)
            weighted_error = (is_weights*td_error).sum()
            #weighted_error = td_error.sum()
            weighted_error.backward() 
            self.optimizer.step()
        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(weighted_error.item())
        return td_error.cpu().detach().numpy()


class NStepDQN(DQN):
    def __init__(self, state_format, actions_number, nstep=1, gamma=0.99, learning_rate=0.00025, ddqn=False, **kwargs):
        self.inputs = dict()
        self.learning_rate = learning_rate
        architecture = kwargs

        self.loss_history = []
        self.misc_state_included = (state_format["s_misc"] > 0)
        self.gamma = np.float64(gamma)
        self.nstep = nstep
        
        #https://pytorch.org/docs/stable/tensors.html
        self.inputs["S0"] = torch.tensor([])
        self.inputs["S1"] = torch.tensor([])
        self.inputs["A"] = torch.tensor([])
        self.inputs["R"] = torch.tensor([])
        self.inputs["Nonterminal"] = torch.tensor([])
        if self.misc_state_included:
            self.inputs["S0_misc"] = torch.tensor([])
            self.inputs["S1_misc"] = torch.tensor([])
            self.misc_len = state_format["s_misc"]
        else:
            self.misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self.single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self.misc_len
        architecture["output_size"] = actions_number

        if self.misc_state_included:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"],
                                                                     misc_input=self.inputs["S0_misc"],
                                                                     **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                misc_input=self.inputs["S1_misc"],
                                                                                **architecture)
        else:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"], **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                **architecture)

        self.alternate_input_mappings = {}
        for layer, input in zip(input_layers, alternate_inputs):
            self.alternate_input_mappings[layer] = input

        self.network.apply(init_weights)
        self.frozen_network.apply(init_weights)
        self.criterion = CustomLoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01)

    def learn(self, transitions):
        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]
               
        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)

                #q2 = self.network(s2_img, s2_misc)
                #q2_action_ref = torch.argmax(q2, dim=1)

                #q2_frozen = self.frozen_network(s2_img, s2_misc)
                #q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]
            
            q2_max = torch.max(self.frozen_network(s2_img, s2_misc), dim=1)
            target_q = (r + self.gamma**self.nstep * nonterminal * q2_max[0]).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)

            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            td_error = td_error.sum()
            td_error.backward()
            self.optimizer.step()

        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(td_error.item())


# This hasn't been tested at all.
class CategoricalDQN(DQN):
    def __init__(self, state_format, actions_number, atoms_number, v_min, v_max, 
                 gamma=0.99, learning_rate=0.00025, ddqn=False, **kwargs):
        self.inputs = dict()
        self.learning_rate = learning_rate
        architecture = kwargs

        self.loss_history = []
        self.misc_state_included = (state_format["s_misc"] > 0)
        self.gamma = np.float64(gamma)

        self.num_actions = actions_number
        self.num_atoms = atoms_number
        self.v_min = v_min
        self.v_max = v_max
        
        #https://pytorch.org/docs/stable/tensors.html
        self.inputs["S0"] = torch.tensor([])
        self.inputs["S1"] = torch.tensor([])
        self.inputs["A"] = torch.tensor([])
        self.inputs["R"] = torch.tensor([])
        self.inputs["Nonterminal"] = torch.tensor([])
        if self.misc_state_included:
            self.inputs["S0_misc"] = torch.tensor([])
            self.inputs["S1_misc"] = torch.tensor([])
            self.misc_len = state_format["s_misc"]
        else:
            self.misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self.single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self.misc_len
        #architecture["output_size"] = self.num_actions * self.num_atoms
        architecture["output_size"] = (self.num_actions, self.num_atoms)

        if self.misc_state_included:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"],
                                                                     misc_input=self.inputs["S0_misc"],
                                                                     **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                misc_input=self.inputs["S1_misc"],
                                                                                **architecture)
        else:

            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"], **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                **architecture)

        self.alternate_input_mappings = {}
        for layer, input in zip(input_layers, alternate_inputs):
            self.alternate_input_mappings[layer] = input

        self.network.apply(init_weights)
        self.frozen_network.apply(init_weights)
        self.criterion = CustomLoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01)

# INITIALIZE
    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]

        if self.misc_state_included:
            inputs.append(misc_input)
        
        network = ActionCategoricalNetwork(img_input_shape, output_size, dueling=False, noisy=False).to(DEVICE)
        return network, input_layers, inputs

# GET ACTION VALUES
    def estimate_best_action(self, state):
    
    # This goes in get_action_values.
    #def act(self, state):
    #    state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
    #    dist = self.forward(state).data.cpu()
    #    dist = dist * torch.linspace(Vmin, Vmax, num_atoms)
    #    action = dist.sum(2).max(1)[1].numpy()[0]
    #
    #    return action

        if self.misc_state_included:
            state_img = torch.from_numpy(state[0].reshape(self.single_image_input_shape)).float().to(DEVICE)
            state_misc = torch.from_numpy(state[1].reshape(1, self.misc_len)).float().to(DEVICE)
            #qvals = self.network(state_img,  state_misc)
            # Try adding .data.cpu() as above. Maybe it increases speed.
            q_dist = self.network(state_img, state_misc)
            q_dist = q_dist * torch.linspace(self.v_min, self.v_max, self.num_atoms).to(DEVICE)
            
            #a = torch.argmax(qvals).item()
            a = torch.argmax(q_dist.sum(2)).item()
            #a = q_dist.sum(2).max(1)[1].numpy()[0]
        else:
            state_img = torch.from_numpy(state[0].reshape(self.single_image_input_shape)).float().to(DEVICE)
            #qvals = self.network(state_img)
            q_dist = self.network(state_img)
            q_dist = q_dist * torch.linspace(self.v_min, self.v_max, self.num_atoms).to(DEVICE)

            #a = torch.argmax(qvals).item()
            a = torch.argmax(q_dist.sum(2)).item()
            #a = q_dist.sum(2).max(1)[1].numpy()[0]

        return a

    def projection_distribution(self, s2_img, s2_misc, r, nonterminal):
        batch_size = s2_img.size(0)

        delta_z = float(self.v_max - self.v_min)/(self.num_atoms - 1)
        support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        q2_dist = self.frozen_network(s2_img, s2_misc).data.cpu() * support
        #q2_dist = self.frozen_network(s2_img, s2_misc).to(DEVICE) * support
        a2 = q2_dist.sum(2).max(1)[1]
        #print("a2")
        #print(a2)
        #print("q2_dist")
        #print(q2_dist.size())
        a2 = a2.unsqueeze(1).unsqueeze(1).expand(q2_dist.size(0), 1, q2_dist.size(2))
        q2_dist = q2_dist.gather(1, a2).squeeze(1)

        r = r.unsqueeze(1).expand_as(q2_dist)
        nonterminal = nonterminal.unsqueeze(1).expand_as(q2_dist)
        support = support.unsqueeze(0).expand_as(q2_dist)

        Tz = r.data.cpu() + nonterminal.data.cpu()*self.gamma*support
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        b = (Tz - self.v_min)/delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.num_atoms)

        q2_proj_dist = torch.zeros(q2_dist.size())
        q2_proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (q2_dist*(u.float() - b)).view(-1))
        q2_proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (q2_dist*(b - l.float())).view(-1))

        #return q2_proj_dist
        return F.softmax(q2_proj_dist)

# OPTIMIZE
    def learn(self, transitions):
        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]
               
        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)

            ##q = self.network(s1_img, s1_misc)
            ##    #q2 = self.network(s2_img, s2_misc)
            ##    #q2_action_ref = torch.argmax(q2, dim=1)
            ## 
            ##    #q2_frozen = self.frozen_network(s2_img, s2_misc)
            ##    #q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]
            ##
            ##q2_max = torch.max(self.frozen_network(s2_img, s2_misc), dim=1)
            ##target_q = (r + self.gamma * nonterminal * q2_max[0]).to(DEVICE)
            ##predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)

            #[batch_size, num_actions, num_atoms]
            batch_size = s1_img.size()[0]
            #proj_dist = projection_distribution(next_state, reward, done)
            proj_q2_dist = self.projection_distribution(s2_img, s2_misc, r, nonterminal).to(DEVICE)
            #dist = current_model(state)
            #action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
            #dist = dist.gather(1, action).squeeze(1)
            #dist.data.clamp_(0.01, 0.99)
            #loss = - (Variable(proj_dist) * dist.log()).sum(1).mean()
            q_dist = self.network(s1_img, s1_misc)
            a = a.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
            q_dist = q_dist.gather(1, a).squeeze(1)
            #q_dist.data.clamp_(0.01, 0.99)

            self.optimizer.zero_grad()
            #td_error = self.criterion(predicted_q, target_q)
            #td_error = td_error.sum()
            #td_error.backward()
            #loss = -(proj_q2_dist * q_dist.log()).sum(1).mean()
            dist_error = (proj_q2_dist * (proj_q2_dist.log() - q_dist.log())).sum(1).mean()
            dist_error.backward()
            self.optimizer.step()
        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        #self.loss_history.append(td_error.item())
        self.loss_history.append(dist_error.item())


# This hasn't been tested at all.
class NoisyDQN(DQN):
# INITIALIZE
    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]

        if self.misc_state_included:
            inputs.append(misc_input)
        
        network = ActionValueNetwork(img_input_shape, output_size, dueling=False, noisy=True).to(DEVICE)
        return network, input_layers, inputs
    
# OPTIMIZE
    def learn(self, transitions):
        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]
               
        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)
                #q2 = self.network(s2_img, s2_misc)
                #q2_action_ref = torch.argmax(q2, dim=1)
            
                #q2_frozen = self.frozen_network(s2_img, s2_misc)
                #q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]
            
            q2_max = torch.max(self.frozen_network(s2_img, s2_misc), dim=1)
            target_q = (r + self.gamma * nonterminal * q2_max[0]).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)
            
            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            td_error = td_error.sum()
            td_error.backward()
            self.optimizer.step()

            self.network.reset_noise()
            self.frozen_network.reset_noise()
        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(td_error.item())


class IntegratedDQN(DQN):
    def __init__(self, state_format, actions_number, nstep=1, gamma=0.99, learning_rate=0.00025, ddqn=False, **kwargs):
        self.inputs = dict()
        self.learning_rate = learning_rate
        architecture = kwargs

        self.loss_history = []
        self.misc_state_included = (state_format["s_misc"] > 0)
        self.gamma = np.float64(gamma)
        self.nstep = nstep
        
        #https://pytorch.org/docs/stable/tensors.html
        self.inputs["S0"] = torch.tensor([])
        self.inputs["S1"] = torch.tensor([])
        self.inputs["A"] = torch.tensor([])
        self.inputs["R"] = torch.tensor([])
        self.inputs["Nonterminal"] = torch.tensor([])
        if self.misc_state_included:
            self.inputs["S0_misc"] = torch.tensor([])
            self.inputs["S1_misc"] = torch.tensor([])
            self.misc_len = state_format["s_misc"]
        else:
            self.misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self.single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self.misc_len
        architecture["output_size"] = actions_number

        if self.misc_state_included:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"],
                                                                     misc_input=self.inputs["S0_misc"],
                                                                     **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                misc_input=self.inputs["S1_misc"],
                                                                                **architecture)
        else:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"], **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                **architecture)

        self.alternate_input_mappings = {}
        for layer, input in zip(input_layers, alternate_inputs):
            self.alternate_input_mappings[layer] = input

        self.network.apply(init_weights)
        self.frozen_network.apply(init_weights)
        self.criterion = CustomLoss()
        #self.criterion = ISLoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01)

    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]

        if self.misc_state_included:
            inputs.append(misc_input)
        
        network = ActionValueNetwork(img_input_shape, output_size, dueling=True, noisy=False).to(DEVICE)
        return network, input_layers, inputs

# OPTIMIZE
    def learn(self, transitions, ISWeights):
        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]

        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)
            is_weights = torch.from_numpy(ISWeights).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)

            #with torch.no_grad():
            q2 = self.network(s2_img, s2_misc)
            q2_action_ref = torch.argmax(q2, dim=1)
            
            q2_frozen = self.frozen_network(s2_img, s2_misc)
            q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]

            target_q = (r + self.gamma**self.nstep * nonterminal * q2_max).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)

            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            #td_error = self.criterion(predicted_q, target_q, is_weights)
            weighted_error = (is_weights*td_error).sum()
            #weighted_error = td_error.sum()
            weighted_error.backward()            
            self.optimizer.step()
        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(weighted_error.item())
        return td_error.cpu().detach().numpy()

class RainbowDQN(CategoricalDQN):
    def __init__(self, state_format, actions_number, atoms_number, v_min, v_max, nstep=1, gamma=0.99, learning_rate=0.00025, ddqn=False, **kwargs):
        self.inputs = dict()
        self.learning_rate = learning_rate
        architecture = kwargs

        self.loss_history = []
        self.misc_state_included = (state_format["s_misc"] > 0)
        self.gamma = np.float64(gamma)
        self.nstep = nstep
        
        self.num_actions = actions_number
        self.num_atoms = atoms_number
        self.v_min = v_min
        self.v_max = v_max

        #https://pytorch.org/docs/stable/tensors.html
        self.inputs["S0"] = torch.tensor([])
        self.inputs["S1"] = torch.tensor([])
        self.inputs["A"] = torch.tensor([])
        self.inputs["R"] = torch.tensor([])
        self.inputs["Nonterminal"] = torch.tensor([])
        if self.misc_state_included:
            self.inputs["S0_misc"] = torch.tensor([])
            self.inputs["S1_misc"] = torch.tensor([])
            self.misc_len = state_format["s_misc"]
        else:
            self.misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self.single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self.misc_len
        #architecture["output_size"] = actions_number
        architecture["output_size"] = (self.num_actions, self.num_atoms)

        if self.misc_state_included:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"],
                                                                     misc_input=self.inputs["S0_misc"],
                                                                     **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                misc_input=self.inputs["S1_misc"],
                                                                                **architecture)
        else:

            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"], **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                **architecture)

        self.alternate_input_mappings = {}
        for layer, input in zip(input_layers, alternate_inputs):
            self.alternate_input_mappings[layer] = input

        self.network.apply(init_weights)
        self.frozen_network.apply(init_weights)
        self.criterion = CustomLoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01)


    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]

        if self.misc_state_included:
            inputs.append(misc_input)
        
        network = ActionCategoricalNetwork(img_input_shape, output_size, dueling=True, noisy=True).to(DEVICE)
        return network, input_layers, inputs
    
    # Identical to the one in CategoricalDQN
    #def projection_distribution(next_state, rewards, dones):
    
    def projection_distribution(self, s2_img, s2_misc, r, nonterminal):
        batch_size = s2_img.size(0)

        delta_z = float(self.v_max - self.v_min)/(self.num_atoms - 1)
        support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        
        #with torch.no_grad():
        q_dist = self.network(s2_img, s2_misc).data.cpu() * support
        #q_dist = self.network(s2_img, s2_misc).to(DEVICE) * support        
        a = q_dist.sum(2).max(1)[1]
        a = a.unsqueeze(1).unsqueeze(1).expand(q_dist.size(0), 1, q_dist.size(2))
        q2_dist = self.frozen_network(s2_img, s2_misc).data.cpu() * support
        q2_dist = q2_dist.gather(1, a).squeeze(1)
        #a2 = q2_dist.sum(2).max(1)[1]
        #a2 = a2.unsqueeze(1).unsqueeze(1).expand(q2_dist.size(0), 1, q2_dist.size(2))
        #q2_dist = q2_dist.gather(1, a2).squeeze(1)

        r = r.unsqueeze(1).expand_as(q2_dist)
        nonterminal = nonterminal.unsqueeze(1).expand_as(q2_dist)
        support = support.unsqueeze(0).expand_as(q2_dist)

        #Tz = r + nonterminal*(self.gamma**self.nstep)*support
        Tz = r.data.cpu() + nonterminal.data.cpu()*(self.gamma**self.nstep)*support
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        b = (Tz - self.v_min)/delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.num_atoms)

        q2_proj_dist = torch.zeros(q2_dist.size())
        q2_proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (q2_dist*(u.float() - b)).view(-1))
        q2_proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (q2_dist*(b - l.float())).view(-1))

        #return q2_proj_dist
        return F.softmax(q2_proj_dist)

# OPTIMIZE
    def learn(self, transitions, ISWeights):
        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]

        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)
            is_weights = torch.from_numpy(ISWeights).float().to(DEVICE)

            #[batch_size, num_actions, num_atoms]
            #proj_dist = projection_distribution(next_state, reward, done)
            #dist = current_model(state)
            #action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
            #dist = dist.gather(1, action).squeeze(1)
            #dist.data.clamp_(0.01, 0.99)
            #loss = - (Variable(proj_dist) * dist.log()).sum(1).mean()      
            batch_size = s1_img.size()[0]
            proj_q2_dist = self.projection_distribution(s2_img, s2_misc, r, nonterminal).to(DEVICE)
            q_dist = self.network(s1_img, s1_misc)
            a = a.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
            q_dist = q_dist.gather(1, a).squeeze(1)
            #q_dist.data.clamp_(0.01, 0.99)
            #proj_q2_dist.data.clamp_(0.01, 0.99)

            #self.optimizer.zero_grad()
            #td_error = self.criterion(predicted_q, target_q)
            ##td_error = self.criterion(predicted_q, target_q, is_weights)
            #weighted_error = (is_weights*td_error).sum()
            ##weighted_error = td_error.sum()
            #weighted_error.backward()            
            #self.optimizer.step()
            self.optimizer.zero_grad()
            dist_error = (proj_q2_dist * (proj_q2_dist.log() - q_dist.log())).sum(1)
            weighted_error = (dist_error * is_weights).mean()
            weighted_error.backward()
            self.optimizer.step()

            self.network.reset_noise()
            self.frozen_network.reset_noise()
        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        #self.loss_history.append(weighted_error.item())
        #return td_error.cpu().detach().numpy()
        self.loss_history.append(weighted_error.item())
        return dist_error.cpu().detach().numpy()

#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

class NDoubleDQN(DQN):
    def __init__(self, state_format, actions_number, nstep=1, gamma=0.99, learning_rate=0.00025, ddqn=False, **kwargs):
        self.inputs = dict()
        self.learning_rate = learning_rate
        architecture = kwargs

        self.loss_history = []
        self.misc_state_included = (state_format["s_misc"] > 0)
        self.gamma = np.float64(gamma)
        self.nstep = nstep
        
        #https://pytorch.org/docs/stable/tensors.html
        self.inputs["S0"] = torch.tensor([])
        self.inputs["S1"] = torch.tensor([])
        self.inputs["A"] = torch.tensor([])
        self.inputs["R"] = torch.tensor([])
        self.inputs["Nonterminal"] = torch.tensor([])
        if self.misc_state_included:
            self.inputs["S0_misc"] = torch.tensor([])
            self.inputs["S1_misc"] = torch.tensor([])
            self.misc_len = state_format["s_misc"]
        else:
            self.misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self.single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self.misc_len
        architecture["output_size"] = actions_number

        if self.misc_state_included:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"],
                                                                     misc_input=self.inputs["S0_misc"],
                                                                     **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                misc_input=self.inputs["S1_misc"],
                                                                                **architecture)
        else:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"], **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                **architecture)

        self.alternate_input_mappings = {}
        for layer, input in zip(input_layers, alternate_inputs):
            self.alternate_input_mappings[layer] = input

        self.network.apply(init_weights)
        self.frozen_network.apply(init_weights)
        self.criterion = CustomLoss()
        #self.criterion = ISLoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01)

    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]

        if self.misc_state_included:
            inputs.append(misc_input)
        
        network = ActionValueNetwork(img_input_shape, output_size, dueling=True, noisy=False).to(DEVICE)
        return network, input_layers, inputs

# OPTIMIZE
    def learn(self, transitions, ISWeights):

        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]
               
        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)
            is_weights = torch.from_numpy(ISWeights).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)

            #with torch.no_grad():
            #q2 = self.network(s2_img, s2_misc)
            #q2_action_ref = torch.argmax(q2, dim=1)
            
            #q2_frozen = self.frozen_network(s2_img, s2_misc)
            #q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]

            q2_max = torch.max(self.frozen_network(s2_img, s2_misc), dim=1)
            target_q = (r + self.gamma**self.nstep * nonterminal * q2_max[0]).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)

            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            #td_error = self.criterion(predicted_q, target_q, is_weights)
            weighted_error = (is_weights*td_error).sum()
            #weighted_error = td_error.sum()
            weighted_error.backward()            
            self.optimizer.step()
        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(weighted_error.item())
        return td_error.cpu().detach().numpy()


class NDuelingDQN(DQN):
    def __init__(self, state_format, actions_number, nstep=1, gamma=0.99, learning_rate=0.00025, ddqn=False, **kwargs):
        self.inputs = dict()
        self.learning_rate = learning_rate
        architecture = kwargs

        self.loss_history = []
        self.misc_state_included = (state_format["s_misc"] > 0)
        self.gamma = np.float64(gamma)
        self.nstep = nstep
        
        #https://pytorch.org/docs/stable/tensors.html
        self.inputs["S0"] = torch.tensor([])
        self.inputs["S1"] = torch.tensor([])
        self.inputs["A"] = torch.tensor([])
        self.inputs["R"] = torch.tensor([])
        self.inputs["Nonterminal"] = torch.tensor([])
        if self.misc_state_included:
            self.inputs["S0_misc"] = torch.tensor([])
            self.inputs["S1_misc"] = torch.tensor([])
            self.misc_len = state_format["s_misc"]
        else:
            self.misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self.single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self.misc_len
        architecture["output_size"] = actions_number

        if self.misc_state_included:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"],
                                                                     misc_input=self.inputs["S0_misc"],
                                                                     **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                misc_input=self.inputs["S1_misc"],
                                                                                **architecture)
        else:

            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"], **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                **architecture)

        self.alternate_input_mappings = {}
        for layer, input in zip(input_layers, alternate_inputs):
            self.alternate_input_mappings[layer] = input

        self.network.apply(init_weights)
        self.frozen_network.apply(init_weights)
        self.criterion = CustomLoss()
        #self.criterion = ISLoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01)

# OPTIMIZE
    def learn(self, transitions, ISWeights):

        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]
               
        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)
            is_weights = torch.from_numpy(ISWeights).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)

            #with torch.no_grad():
            q2 = self.network(s2_img, s2_misc)
            q2_action_ref = torch.argmax(q2, dim=1)

            q2_frozen = self.frozen_network(s2_img, s2_misc)
            q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]

            target_q = (r + self.gamma**self.nstep * nonterminal * q2_max).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)

            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            #td_error = self.criterion(predicted_q, target_q, is_weights)
            weighted_error = (is_weights*td_error).sum()
            #weighted_error = td_error.sum()
            weighted_error.backward()            
            self.optimizer.step()


        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(weighted_error.item())
        return td_error.cpu().detach().numpy()

class NNStepDQN(DQN):
    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]

        if self.misc_state_included:
            inputs.append(misc_input)

        network = ActionValueNetwork(img_input_shape, output_size, dueling=True, noisy=False).to(DEVICE)
        return network, input_layers, inputs

# OPTIMIZE
    def learn(self, transitions, ISWeights):
        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]

        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)
            is_weights = torch.from_numpy(ISWeights).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)

            #with torch.no_grad():
            q2 = self.network(s2_img, s2_misc)
            q2_action_ref = torch.argmax(q2, dim=1)
            
            q2_frozen = self.frozen_network(s2_img, s2_misc)
            q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]

            target_q = (r + self.gamma * nonterminal * q2_max).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)

            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            #td_error = self.criterion(predicted_q, target_q, is_weights)
            weighted_error = (is_weights*td_error).sum()
            #weighted_error = td_error.sum()
            weighted_error.backward()            
            self.optimizer.step()
        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(weighted_error.item())
        return td_error.cpu().detach().numpy()


class NPrioritizedDQN(DQN):
    def __init__(self, state_format, actions_number, nstep=1, gamma=0.99, learning_rate=0.00025, ddqn=False, **kwargs):
        self.inputs = dict()
        self.learning_rate = learning_rate
        architecture = kwargs

        self.loss_history = []
        self.misc_state_included = (state_format["s_misc"] > 0)
        self.gamma = np.float64(gamma)
        self.nstep = nstep
        
        #https://pytorch.org/docs/stable/tensors.html
        self.inputs["S0"] = torch.tensor([])
        self.inputs["S1"] = torch.tensor([])
        self.inputs["A"] = torch.tensor([])
        self.inputs["R"] = torch.tensor([])
        self.inputs["Nonterminal"] = torch.tensor([])
        if self.misc_state_included:
            self.inputs["S0_misc"] = torch.tensor([])
            self.inputs["S1_misc"] = torch.tensor([])
            self.misc_len = state_format["s_misc"]
        else:
            self.misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self.single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self.misc_len
        architecture["output_size"] = actions_number

        if self.misc_state_included:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"],
                                                                     misc_input=self.inputs["S0_misc"],
                                                                     **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                misc_input=self.inputs["S1_misc"],
                                                                                **architecture)
        else:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"], **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                **architecture)

        self.alternate_input_mappings = {}
        for layer, input in zip(input_layers, alternate_inputs):
            self.alternate_input_mappings[layer] = input

        self.network.apply(init_weights)
        self.frozen_network.apply(init_weights)
        self.criterion = CustomLoss()
        #self.criterion = ISLoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01)

    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]

        if self.misc_state_included:
            inputs.append(misc_input)
        
        network = ActionValueNetwork(img_input_shape, output_size, dueling=True, noisy=False).to(DEVICE)
        return network, input_layers, inputs

# OPTIMIZE
    def learn(self, transitions):

        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]
        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]

        t = transitions
        if self.misc_state_included:
            s1_img = torch.from_numpy(t["s1_img"]).float().to(DEVICE)
            s1_misc = torch.from_numpy(t["s1_misc"]).float().to(DEVICE)
            s2_img = torch.from_numpy(t["s2_img"]).float().to(DEVICE)
            s2_misc = torch.from_numpy(t["s2_misc"]).float().to(DEVICE)
            a = torch.from_numpy(t["a"]).long().to(DEVICE)
            r = torch.from_numpy(t["r"]).float().to(DEVICE)
            nonterminal = torch.from_numpy(t["nonterminal"].astype(np.uint8)).float().to(DEVICE)

            q = self.network(s1_img, s1_misc)

            #with torch.no_grad():
            q2 = self.network(s2_img, s2_misc)
            q2_action_ref = torch.argmax(q2, dim=1)
            
            q2_frozen = self.frozen_network(s2_img, s2_misc)
            q2_max = q2_frozen[torch.arange(q2_action_ref.shape[0]), q2_action_ref]

            target_q = (r + self.gamma**self.nstep * nonterminal * q2_max).to(DEVICE)
            predicted_q = (q[torch.arange(q.shape[0]), a]).to(DEVICE)

            self.optimizer.zero_grad()
            td_error = self.criterion(predicted_q, target_q)
            #td_error = self.criterion(predicted_q, target_q, is_weights)
            #weighted_error = (is_weights*td_error).sum()
            weighted_error = td_error.sum()
            weighted_error.backward()            
            self.optimizer.step()

        else:
            #loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
            print("trt")
            exit(1)

        self.loss_history.append(weighted_error.item())
        return td_error.cpu().detach().numpy()

