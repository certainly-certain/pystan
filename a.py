from numpy import inf, array, exp, matrix, diag, multiply, ones, asarray, log
from numpy.random import randint
from numpy.linalg import norm, solve

def search_sigma_and_lambda(x, y, alpha, centers, sigma_range, lambda_range):
    nx = x.shape[0]
    ny = y.shape[0]
    n_min = min(nx, ny)
    kernel_num = centers.shape[0]

    score_new = inf
    sigma_new = 0
    lambda_new = 0

    for sigma in sigma_range:

        phi_x = compute_kernel_Gaussian(x, centers, sigma)
        print(f"phi_x.shape:{phi_x.shape}")
        phi_y = compute_kernel_Gaussian(y, centers, sigma)
        print(f"phi_y.shape:{phi_y.shape}")
        H = alpha * (phi_x.T.dot(phi_x) / nx) + (1 - alpha) * (phi_y.T.dot(phi_y) / ny)
        print(f"H.shape:{H.shape}")
        h = phi_x.mean(axis=0).T
        print(f"h.shape:{h.shape}")
        phi_x = phi_x[:n_min].T
        print(f"phi_x.shape:{phi_x.shape}")
        phi_y = phi_y[:n_min].T
        print(f"phi_y.shape:{phi_y.shape}")

        for lambda_ in lambda_range:
            B = H + diag(array(lambda_ * (ny - 1) / ny).repeat(kernel_num))
            print(f"array(lambda_ * (ny - 1) / ny).repeat(kernel_num):")
            print(array(lambda_ * (ny - 1) / ny).repeat(kernel_num))
            print(f"diag(array(lambda_ * (ny - 1) / ny).repeat(kernel_num)):")
            print(diag(array(lambda_ * (ny - 1) / ny).repeat(kernel_num)))
            print(f"B.shape:{B.shape}")
            B_inv_X = solve(B, phi_y)
            print(f"B_inv_X.shape:{B_inv_X.shape}")
            X_B_inv_X = multiply(phi_y, B_inv_X)
            print(f"X_B_inv_X.shape:{X_B_inv_X.shape}")
            denom = (ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X)).A1
            print(f"ny * ones(n_min):")
            print(ny * ones(n_min))
            print(f"ones(kernel_num).dot(X_B_inv_X):")
            print(ny * ones(n_min))
            print(f"ones(kernel_num).dot(X_B_inv_X):")
            print(ones(kernel_num).dot(X_B_inv_X))
            print(f"ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X):")
            print(ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X))
            print(f"(ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X)).A1:")
            print((ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X)).A1)
            print(f"denom.shape:{denom.shape}")
            B0 = solve(B, h.dot(matrix(ones(n_min)))) + B_inv_X.dot(diag(h.T.dot(B_inv_X).A1 / denom))
            print(f"ones(kernel_num).dot(X_B_inv_X):")
            print(ones(kernel_num).dot(X_B_inv_X))
            print(f"ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X):")
            print(ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X))
            print(f"(ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X)).A1:")
            print((ny * ones(n_min) - ones(kernel_num).dot(X_B_inv_X)).A1)
            print(f"denom.shape:{denom.shape}")
            B1 = solve(B, phi_x) + B_inv_X.dot(diag(ones(kernel_num).dot(multiply(phi_x, B_inv_X)).A1))
            B2 = (ny - 1) * (nx * B0 - B1) / (ny * (nx - 1))
            B2[B2 < 0] = 0
            r_y = multiply(phi_y, B2).sum(axis=0).T
            r_x = multiply(phi_x, B2).sum(axis=0).T

            # Squared loss of RuLSIF, without regularization term.
            # Directly related to the negative of the PE-divergence.
            score = (r_y.T.dot(r_y).A1 / 2 - r_x.sum(axis=0)) / n_min

            if score < score_new:
                score_new = score
                sigma_new = sigma
                lambda_new = lambda_

        print("\n\n")

    return {"sigma": sigma_new, "lambda": lambda_new}


# Returns a 2D numpy matrix of kernel evaluated at the gridpoints with coordinates from x_list and y_list.
def compute_kernel_Gaussian(x_list, y_list, sigma):
    result = [[kernel_Gaussian(x, y, sigma) for y in y_list] for x in x_list]
    result = matrix(result)
    return result


# Returns the Gaussian kernel evaluated at (x, y) with parameter sigma.
def kernel_Gaussian(x, y, sigma):
    return exp(- (norm(x - y) ** 2) / (2 * sigma * sigma))
    
#params
alpha = 0
centers = np.array([1.,2.])
sigma_range = np.array([1,2])
lambda_range = np.array([1,2])

x = np.arange(8).reshape(4,2)
x

y = np.arange(6).reshape(3,2)
y

search_sigma_and_lambda(x, y, alpha, centers, sigma_range, lambda_range)
