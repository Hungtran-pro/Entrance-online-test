from lib import *

random_state = 42
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)