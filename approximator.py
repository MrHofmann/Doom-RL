from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


class DuelingActionValueNetwork(nn.Module):
    def __init__(self, img_input_shape, output_size, ddqn=False):
        super(DuelingActionValueNetwork, self).__init__()
        
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
       
        #self.out = nn.Sequential(nn.Linear(512, output_size, bias=True))


    def forward(self, x, misc=None): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = torch.cat((x.view(-1, 3136), misc), dim=1)
        ##x = self.fc1(x)
        ##x = self.out(x)
        #x1 = x[:, :1586]
        #x2 = x[:, 1586:]
        #state_value = self.state_fc(x1).reshape(-1, 1)
        #advantage_values = self.advantage_fc(x2)
        #x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        x = torch.cat((x.view(x.size(0), -1), misc), dim=1)
        state_value = self.state_fc(x)
        advantage_values = self.advantage_fc(x)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        #print(x)
        return x


class ActionValueNetwork(nn.Module):
    def __init__(self, img_input_shape, output_size, ddqn=False):
        super(ActionValueNetwork, self).__init__()

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
        self.fc1 = nn.Sequential(
            nn.Linear(3172, 512, bias=True),
            nn.ReLU()
            )
        
        self.out = nn.Sequential(nn.Linear(512, output_size, bias=True))

    def forward(self, x, misc=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
        
        network = ActionValueNetwork(img_input_shape, output_size).to(DEVICE)
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
        
        network = DuelingActionValueNetwork(img_input_shape, output_size).to(DEVICE)
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
        
        network = DuelingActionValueNetwork(img_input_shape, output_size).to(DEVICE)
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
        
        network = DuelingActionValueNetwork(img_input_shape, output_size).to(DEVICE)
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

        network = DuelingActionValueNetwork(img_input_shape, output_size).to(DEVICE)
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
        
        network = DuelingActionValueNetwork(img_input_shape, output_size).to(DEVICE)
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

