from rmdp import *
import matplotlib.pyplot as plt

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
        p_0 = np.array([1,0])
        gamma = 0.99
        reward = np.array([1,0,0,0])
        super().__init__(num_states, num_actions, num_features, P, phi, p_0, gamma, reward)

def get_U_xi(xi, gamma):
    return np.array([(xi/(1-gamma)), (gamma/(1-gamma))*(1-xi), (1-xi), 0])

def plot_lpal_error(env: MDP):
    D = [(0,0), (1,0)]
    u_hat = (1/(1-env.gamma))*np.array([0.5,0.5,0,0])
    xi_list = np.arange(0.0, 1.0, 0.001)
    linf_error = []
    for xi in xi_list:
        u_xi = get_U_xi(xi, env.gamma)
        linf_error.append(np.linalg.norm(u_xi - u_hat, ord=np.inf))
    plt.plot(xi_list, linf_error, label="LPAL Loss")
    plt.xlabel("Xi")
    plt.ylabel("||u_xi - u_hat||_inf")
    plt.title(f"LPAL Loss vs Xi")
    # Move legend to outside of plot
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"plots/all_state/lpal_loss.pdf")
    plt.clf()

    

def main():
    env = TwoState()
    print(f"Opt policy = {env.opt_policy}")
    print(f"Opt occ freq = {env.u_E}")
    print(f"Opt value = {env.opt_return}")
    # Set the font type to TrueType Globally
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # set the font to be Computer Modern (cmr10 doesnt work so we use serif)
    plt.rcParams["font.family"] = "serif"
    plot_lpal_error(env)

if __name__ == "__main__":
    main()
