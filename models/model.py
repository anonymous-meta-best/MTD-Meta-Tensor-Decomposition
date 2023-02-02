"""
Meta-Learning for Fast and Accurate Domain Adaptation for Irregular Tensors
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyun.park@snu.ac.kr), Seoul National University
- Jun-Gi Jang (elnino4@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""

from torch import nn
from .utils import *

class meta_decomposition(nn.Module):
    """
    Meta Tensor Decomposition
    """
    def __init__(self, n_sensors, rank, time_length, max_time_length, n_tasks, missing_idxs, task, imputation, residual):
        """
        initializer
        """
        super(meta_decomposition, self).__init__()

        n_instances_all = len(time_length)
        n_instances = int(len(time_length) / n_tasks)

        self.Q = nn.Parameter(torch.rand(n_instances_all, max_time_length, rank, dtype=torch.float32))
        for idx, time_length in enumerate(time_length):
            self.Q.data[idx, time_length:] = 0
        self.Q = nn.Parameter(self.Q.view(n_tasks, n_instances, max_time_length, rank))

        self.W = nn.Parameter(torch.rand(n_tasks, n_instances, rank, dtype=torch.float32))
        self.V = nn.Parameter(torch.rand(n_sensors, rank, dtype=torch.float32))
        self.V_meta = nn.Parameter(torch.rand(n_sensors, rank, dtype=torch.float32))
        self.H = nn.Parameter(torch.rand(n_tasks, rank, rank, dtype=torch.float32))

        self.n_tasks = n_tasks
        self.n_instances = n_instances
        self.missing_idxs = missing_idxs
        self.task = task
        self.imputation = imputation
        self.residual = residual

    def forward(self):
        """
        Return the diagonal matrix
        """
        return self.W[:-1], self.W[-1].unsqueeze(0)

    def train_model(self, X, X_true, epochs):
        """
        train model
        """
        with torch.no_grad():
            print("[Decomposition on Source Domains]")

            for epoch in range(1, epochs+1):
                X_pred = self.loop(X)

                if epoch==1 or epoch % 10 == 0:
                    rec_err = torch.norm(X_pred[:-1] - X_true[:-1])
                    recovery_rate = 1 - rec_err / torch.norm(X_true[:-1])
                    print("Epoch: {} \tReconstruction Rate: {:.4f}".format(epoch, recovery_rate))

            print("\n[Decomposition on Target Domain]")
            self.V_meta = nn.Parameter(self.V.data)

            rec_errs = []
            for epoch in range(1, epochs+1):
                X_pred = self.loop(X, True)

                if self.imputation == "y":
                    if epoch == 1:
                        X[-1] = self.update_missing_values(X[-1], X_pred[-1])
                missing_err = self.compute_missing_rate(X_true[-1], X_pred[-1])

                rec_err = torch.norm(X_pred[-1] - X_true[-1])

                if epoch==1 or epoch % 10 == 0:
                    recovery_err = 1 - rec_err / torch.norm(X_true[-1])
                    print("Epoch: {} \tReconstruction Rate: {:.4f}".format(epoch, recovery_err))
                rec_errs.append(missing_err.item())

            return rec_errs

    def _update_Q(self, X, dom):
        """
        update Q with truncated SVD
        """
        XV = torch.einsum("bij, jk -> bik", X[dom], self.V)

        S = torch.stack([torch.diag(s) for s in self.W[dom]])
        XVS = torch.einsum("bij, bjk -> bik", XV, S)
        XVSH_t = torch.einsum("bij, jk -> bik", XVS, self.H[dom].T)

        Z, S_d, P = torch.svd(XVSH_t)
        P_t = P.permute(0, 2, 1)

        return torch.einsum("bij, bjk -> bik", Z, P_t)

    def _update_HVW(self, Y, domain, is_test=False):
        """
        update H, V, and W
        """
        Y1 = Y.permute(0, 2, 1).reshape(Y.size(0), -1)
        Y2 = Y.permute(2, 0, 1).reshape(-1, Y.size(1)).T
        Y3 = Y.permute(1, 0, 2).reshape(-1, Y.size(2)).T

        H_inv = torch.matmul(self.W[domain].T, self.W[domain]) * torch.matmul(self.V.T, self.V)
        self.H[domain] = nn.Parameter(torch.matmul(torch.matmul(Y1, khatri(self.W[domain], self.V)), torch.pinverse(H_inv)))
        W_inv = torch.matmul(self.V.T, self.V) * torch.matmul(self.H[domain].T, self.H[domain])
        self.W[domain] = nn.Parameter(torch.matmul(torch.matmul(Y3, khatri(self.V, self.H[domain])), torch.pinverse(W_inv)))
        V_inv = torch.matmul(self.W[domain].T, self.W[domain]) * torch.matmul(self.H[domain].T, self.H[domain])
        self.V = nn.Parameter(torch.matmul(torch.matmul(Y2, khatri(self.W[domain], self.H[domain])), torch.pinverse(V_inv)))

        if is_test:
            if self.residual == 'y':
                return (self.V + self.V_meta)/2
            else:
                return self.V
        else:
            H_inv = torch.matmul(self.W[domain].T, self.W[domain]) * torch.matmul(self.V.T, self.V)
            H1 = nn.Parameter(torch.matmul(torch.matmul(Y1, khatri(self.W[domain], self.V)), torch.pinverse(H_inv)))
            W_inv = torch.matmul(self.V.T, self.V) * torch.matmul(H1.T, H1)
            W1 = nn.Parameter(torch.matmul(torch.matmul(Y3, khatri(self.V, H1)), torch.pinverse(W_inv)))

            V_inv = torch.matmul(W1.T, W1) * torch.matmul(H1.T, H1)
            V2 = torch.matmul(torch.matmul(Y2, khatri(W1, H1)), torch.pinverse(V_inv))
            return V2

    def inner_loop(self, X, domain, is_test=False):
        """
        inner-loop of meta-learning
        """
        self.Q[domain] = nn.Parameter(self._update_Q(X, domain))
        Y = torch.einsum("bij, bjk -> bik", self.Q[domain].permute((0, 2, 1)), X[domain]).permute((1, 2, 0))
        V_t = self._update_HVW(Y, domain, is_test)

        return V_t

    def loop(self, X, is_test=False):
        """
        outer-loop of meta-learning
        """
        if is_test:
            self.V = nn.Parameter(self.inner_loop(X, -1, is_test))
        else:
            n_trn_tasks = self.n_tasks-1

            V = 0
            for i in range(n_trn_tasks):
                V += self.inner_loop(X, i)
            self.V = nn.Parameter(V/n_trn_tasks)
        return self.reconstruct()

    def reconstruct(self):
        """
        reconstruct the original tensor with computed latent factor matrices
        """
        Q = self.Q.flatten(0, 1)
        W = self.W.flatten(0, 1)
        H = torch.cat([h.repeat([self.n_instances, 1, 1]) for h in self.H])

        U = torch.einsum("bij, bjk -> bik", Q, H)
        S = torch.stack([torch.diag(s) for s in W])
        US = torch.einsum("bij, bjk -> bik", U, S)
        X_pred = torch.einsum("bij, jk -> bik", US, self.V.T)

        return X_pred.view(self.n_tasks, -1, X_pred.size(1), X_pred.size(2))

    def compute_missing_rate(self, X, X_pred):
        """
        compute the normalized missing value prediction error
        """
        x = X.permute((0, 2, 1))
        x_pred = X_pred.permute((0, 2, 1))

        err, norm = 0, 0
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                idx = self.missing_idxs[i*x.size(1) + j]
                err += torch.pow(torch.norm(x_pred[i,j][idx] - x[i,j][idx]), 2)
                norm += torch.pow(torch.norm(x[i,j][idx]), 2)

        return err / norm

    def update_missing_values(self, X, X_pred):
        """
        Imputation on the target domain
        """
        x = X.permute((0, 2, 1))
        x_pred = X_pred.permute((0, 2, 1))

        for i in range(x.size(0)):
            for j in range(x.size(1)):
                idx = self.missing_idxs[i*x.size(1) + j]
                x[i,j][idx] = x_pred[i,j][idx]

        return x.permute((0, 2, 1))