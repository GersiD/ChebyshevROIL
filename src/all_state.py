from rmdp import *
import matplotlib.pyplot as plt
from scipy import special

class TwoState(MDP):
    def __init__(self):
        num_states = 2
        num_actions = 2
        num_features = 2
        P = np.zeros((num_states, num_states, num_actions))
        # P[s,s',a] = P(s'|s,a)
        P[0,0,0] = 1.0
        P[0,0,1] = 0.0
        P[0,1,0] = 0.0
        P[0,1,1] = 1.0
        P[1,0,0] = 0.0
        P[1,0,1] = 0.0
        P[1,1,0] = 1.0
        P[1,1,1] = 1.0
        for s in range(num_states):  # ensure P is a transition probablity matrix
            for a in range(num_actions):
                assert(sum(P[s, :, a]) - 1 < 1e-10)
        phi = np.zeros((num_states*num_actions, num_features))
        p_0 = np.array([0.9,0.1])
        gamma = 0.99
        reward = np.array([0,1,0,0])
        super().__init__(num_states, num_actions, num_features, P, phi, p_0, gamma, reward)

def get_U_xi(xi, gamma):
    return np.array([(xi/(1-gamma)), (gamma/(1-gamma))*(1-xi), (1-xi), 0])

def JSD(p, q):
    """Compute the Jensen-Shannon Divergence between two probability distributions"""
    m = 0.5 * (p + q)
    return 0.5 * (special.kl_div(p, m) + special.kl_div(q, m)).sum()

def generate_losses(env: MDP):
    # The following assumes the following Dataset
    D = [(0,0), (1,0)]
    u_hat = (1/(1-env.gamma))*np.array([0.5,0.5,0,0])
    xi_list = np.arange(0.0, 1.0, 0.001)
    linf_error = np.zeros(len(xi_list))
    djs_list = np.zeros(len(xi_list))
    for i, xi in enumerate(xi_list):
        u_xi = get_U_xi(xi, env.gamma)
        linf_error[i] = np.linalg.norm(u_xi - u_hat, ord=np.inf)
        djs_list[i] = JSD((1/(1-env.gamma))*u_xi, (1/(1-env.gamma))*u_hat)
    return xi_list, linf_error, djs_list

def plot_lpal_error(env: MDP):
    xi_list, linf_error, _ = generate_losses(env)
    plt.plot(xi_list, linf_error, label="LPAL Loss")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$||u_\xi - \hat{u}_e||_\infty$")
    plt.title(f"LPAL Loss")
    # Move legend to outside of plot
    plt.legend(loc="upper center")
    plt.grid()
    plt.savefig(f"plots/all_state/lpal_loss.pdf")
    plt.clf()

def plot_gail_error(env: MDP):
    xi_list, _, djs_list = generate_losses(env)
    plt.plot(xi_list, djs_list, label="GAIL Loss")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"JSD($u_\xi - \hat{u}_e$)")
    plt.title(f"GAIL Loss")
    # Move legend to outside of plot
    plt.legend(loc="upper center")
    plt.grid()
    plt.savefig(f"plots/all_state/gail_loss.pdf")
    plt.clf()

def main():
    env = TwoState()
    print(f"Opt policy = {env.opt_policy}")
    print(f"Opt occ freq = {env.u_E}")
    print(f"Opt value = {env.opt_return}")
    # # Set the font type to TrueType Globally
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # # set the font to be Computer Modern (cmr10 doesnt work so we use serif)
    plt.rcParams["font.family"] = "serif"
    # increase font size
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(6.7, 5.1))
    plt.rc('text', usetex=True)
    plot_lpal_error(env)
    plot_gail_error(env)

if __name__ == "__main__":
    main()
