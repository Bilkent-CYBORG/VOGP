from algorithms.ADELGP.ADELGP.phases import *
from algorithms.ADELGP.ADELGP.Hyperrectangle import Hyperrectangle
from algorithms.ADELGP.ADELGP.DesignPoint import DesignPoint, DesignPointAD
from models.gpyt import GPModel

from utils.utils import get_max_variance_point, compute_ustar_scipy, normalize


class VectorEpsilonPAL:
    def __init__(
        self, problem_model, cone, epsilon, delta, conf_contraction, is_adaptive,
        unknown_params, init_design_cnt, rkhs_bound,
        maxiter=None, batch_size: int = False
    ):
        self.is_adaptive = is_adaptive
        self.depth_max = (
            None if not is_adaptive else problem_model.dataset.max_discretization_depth
        )
        self.unknown_params = unknown_params
        self.rkhs_bound = rkhs_bound

        self.problem_model = problem_model
        self.d = self.problem_model.dataset.in_dim
        self.m = self.problem_model.dataset.out_dim
        self.cone = cone
        self.epsilon = epsilon
        self.delta = delta
        self.conf_contraction = conf_contraction
        self.maxiter = maxiter
        self.batch_size = batch_size
        # Rounds
        self.t = 0  # Total number of iterations

        self.sample_count = 0

        # Sets
        self.P = []  # Decided pareto design points
        self.D = []
        if not self.is_adaptive:
            self.S = [
                DesignPoint(
                    row,
                    Hyperrectangle(upper=[np.inf] * self.m, lower=[-np.inf] * self.m),
                    design_index=i
                ) for i, row in enumerate(problem_model.x)
            ]  # Undecided design points
        else:
            self.S = [
                DesignPointAD(
                    np.array([0.5] * self.d),
                    Hyperrectangle(upper=[np.inf] * self.m, lower=[-np.inf] * self.m),
                    design_index=1, depth=1, cell_bound=[[0, 1] for _ in range(self.d)]
                )
            ]

        # Create GP model and handle initial data
        self.gp = GPModel(
            input_dim=problem_model.dataset.in_dim, output_dim=problem_model.dataset.out_dim,
            noise_var=problem_model.obs_noise_var, kernel=problem_model.dataset.model_kernel,
            constant_mean=self.is_adaptive
        )
        if self.unknown_params:
            if is_adaptive:
                dimension_sizes = np.array([bound[1] - bound[0] for bound in problem_model.dataset.bounds])
                dimension_lower_bounds = np.array([bound[0] for bound in problem_model.dataset.bounds])
                initial_points = np.random.rand(
                    init_design_cnt, problem_model.dataset.in_dim
                ) * dimension_sizes - dimension_lower_bounds
                initial_values = problem_model.dataset.evaluate(initial_points)
            else:
                for init_i in range(init_design_cnt):
                    if init_i == 0:
                        to_observe = int(np.random.choice(len(problem_model.x), 1))
                    else:
                        to_observe = int(get_max_variance_point(self.gp, problem_model.x))
                    pt_x = problem_model.x[to_observe:to_observe+1]
                    pt_y = problem_model(self.S[to_observe])
                    self.gp.add_sample(pt_x, pt_y)
                    self.gp.update()
                    self.gp.train()
            self.gp.train()
        else:
            if is_adaptive:
                dimension_sizes = np.array(
                    [bound[1] - bound[0] for bound in problem_model.dataset.bounds]
                )
                dimension_lower_bounds = np.array(
                    [bound[0] for bound in problem_model.dataset.bounds]
                )
                initial_points = np.random.rand(
                    problem_model.dataset.domain_discretization_each_dim**problem_model.dataset.in_dim,
                    problem_model.dataset.in_dim
                ) * dimension_sizes + dimension_lower_bounds
                initial_values = problem_model.dataset.evaluate(initial_points)
                self.gp.add_sample(
                    normalize(initial_points, problem_model.dataset.bounds),
                    initial_values
                )
                self.gp.update()
                self.gp.train()
                self.gp.clear_data()

                initial_points = np.random.rand(
                    init_design_cnt, problem_model.dataset.in_dim
                ) * dimension_sizes + dimension_lower_bounds
                initial_values = problem_model.dataset.evaluate(initial_points)
                initial_points = normalize(initial_points, problem_model.dataset.bounds)
            else:
                self.gp.add_sample(problem_model.x, problem_model.y)
                self.gp.update()
                self.gp.train()
                self.gp.clear_data()

                initial_indices = np.random.choice(len(problem_model.x), init_design_cnt)
                initial_points = problem_model.x[initial_indices]
                initial_values = problem_model([self.S[i] for i in initial_indices])
            self.gp.add_sample(initial_points, initial_values)
            self.gp.update()
        self.sample_count += init_design_cnt

        self.beta = np.ones(self.m, )

        self.u_star = self.epsilon * compute_ustar_scipy(self.cone.A)[0]

    def finished(self):
        return len(self.S) == 0

    def run_one_step(self):
        print(f"Round {self.t}")
        # Active nodes, union of sets s_t and p_t at the beginning of round t
        A = self.P + self.S

        print("Modeling")
        # Set beta for this round
        self.beta = self.find_beta()
        modeling(A, self.gp, self.beta)

        print("Discarding")
        discard(self.S, self.P, self.D, self.cone, self.epsilon, self.u_star)

        print("epsilon-Covering")
        # The union of sets S and P at the beginning of epsilon-Covering
        W = self.S + self.P
        epsiloncovering(self.S, self.P, self.cone, self.epsilon, self.depth_max, self.u_star)

        print("Evaluating")
        if self.S:  # If S_t is not empty
            if not self.is_adaptive:
                to_observe_list = evaluate(W, self.gp, self.beta, self.batch_size)
            else:
                to_observe_list = evaluating_refining_AD(
                    W, self.S, self.P, self.gp, self.beta, self.depth_max
                )
            self.sample_count += len(to_observe_list)
            for design in to_observe_list:
                y = self.problem_model(design)
                self.gp.add_sample(design.x.reshape(-1, self.d), y.reshape(-1, self.m))
                self.gp.update()
            if self.unknown_params:
                self.gp.train()
        
        print(
            f"There are {len(self.S)} designs left in set S and"
            f" {len(self.P)} designs in set P."
        )

        if self.S and self.unknown_params:
            self.S = self.D + self.S + self.P
            self.D.clear()
            self.P.clear()

        if self.t == self.maxiter:
            return self.P

        self.t += 1

    def find_beta(self):
        if self.is_adaptive:
            beta_sqr =  2 * np.log(
                2 * self.m * (4**(self.depth_max + 1)) * (np.pi ** 2) * (
                    (self.sample_count+1) ** 2
                ) / (3 * self.delta)
            )

            return np.sqrt(beta_sqr / self.conf_contraction) * np.ones(self.m, )

        # This is according to the proofs.
        beta_sqr = 2 * np.log(
            self.m * self.problem_model.cardinality * (np.pi**2) * ((self.t+1)**2) / (3 * self.delta)
        )

        if self.rkhs_bound:
            Kn = self.gp.model.covar_module(self.gp.X_T, self.gp.X_T).evaluate()
            rkhs_bound = 0.1
            beta_sqr = rkhs_bound + np.sqrt(
                self.problem_model.obs_noise_var * np.log(
                    (1 / self.problem_model.obs_noise_var)
                    * torch.det(Kn + torch.eye(len(Kn))).detach().cpu().numpy()
                ) - 2 * np.log(self.delta)
            )
            beta_sqr = beta_sqr**2

        return np.sqrt(beta_sqr / self.conf_contraction) * np.ones(self.m, )
