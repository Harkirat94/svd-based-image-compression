import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


def display_image(arg_img):
    """method to display the image"""
    cv2.imshow('image', arg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def im2double(im):
    """method to get double precision of a channel"""
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max


def channel_svd(channel):
    """
    method to calculate svd of the input data matrix
    :param channel: data matrix whose svd is to be calculatec
    :return: list of three matrices: U, Sigma and V transpose
    """
    [u, sigma, vt] = np.linalg.svd(channel)
    return [u, sigma, vt]


def calculate_frobenius_norm(channel, u, s_diagonalized, vt):
    """
    method to calculate the frobenius norm of the reconstructed matrix
    Done by taking 1 to k (k=rank of the channel) components in the reconstruction process.
    :return: list of frobenius norms calculated by the first k signular values of the matrix
    """
    rank_channel = np.linalg.matrix_rank(channel)
    frobenius_norm_array = np.zeros(rank_channel)
    for k in range(1, rank_channel+1):
        u_k = u[:, :k]
        s_diagonalized_k = s_diagonalized[:k, :k]
        vt_k = vt[:k, :]
        channel_reconstruction_matrix = np.dot(np.dot(u_k, s_diagonalized_k), vt_k)
        rec_error_matrix = np.subtract(channel, channel_reconstruction_matrix)
        frobenius_norm = np.linalg.norm(rec_error_matrix, 'fro')
        frobenius_norm_array[k-1] = frobenius_norm
    return frobenius_norm_array


def calculate_optimal_k(singular_values, rank):
    """method to calculate the optimal k"""
    heuristic_k = 0.15 * rank
    optimal_k = heuristic_k
    end_idx = len(singular_values) - 10
    for idx in range(0, end_idx):
        sig_diff = max(singular_values[idx:idx + 10]) - min(singular_values[idx:idx + 10])
        if sig_diff < 0.32:
            optimal_k = idx
            break
    return min(heuristic_k, optimal_k)


def channel_via_optimal_k(k, u, s_diagonalized, vt):
    """reconstructs a matrix by selecting optimal_k signular values"""
    channel_u_k = u[:, :k]
    channel_s_diagonal_k = s_diagonalized[:k, :k]
    channel_vt_k = vt[:k, :]
    channel_reconstruction_matrix = np.dot(np.dot(channel_u_k, channel_s_diagonal_k), channel_vt_k)
    channel_reconstruction_matrix = 255 * channel_reconstruction_matrix
    return channel_reconstruction_matrix


def plot_singular_values(s_red, s_blue, s_green):
    """Plot the singular values of the three channels on loglog graph for analysis"""
    non_zero_s_red = s_red[np.nonzero(s_red)]
    non_zero_s_blue = s_blue[np.nonzero(s_blue)]
    non_zero_s_green = s_green[np.nonzero(s_green)]
    x = np.linspace(0, 1, len(non_zero_s_red))
    plt.loglog(x, non_zero_s_red, 'r')
    plt.loglog(x, non_zero_s_blue, 'b')
    plt.loglog(x, non_zero_s_green, 'g')
    plt.ylabel('Singular values of the three channels.')
    plt.show()


def plot_frobenius_norm(red_frobenius_norm, blue_frobenius_norm, green_frobenius_norm, rank_channel):
    """Plot the frobenius norm of the three channels on for analysis"""
    x = np.arange(1, rank_channel+1)
    plt.xlabel('k')
    plt.ylabel('Frobenius norm')
    plt.plot(x, red_frobenius_norm, 'r', x, blue_frobenius_norm, 'b', x, green_frobenius_norm, 'g')
    plt.show()


def percentage_of_info_restored(k, all_singular_values):
    """calculate percentage of information restored. Measured as sum of singular values
    retained in the optimal_k dimensions divided by total sum of singular values."""
    sum_sigma = 0
    sum_k_sigma = 0
    for sigma_matrix in all_singular_values:
        sum_sigma += np.trace(sigma_matrix)
    for sigma_matrix in all_singular_values:
        stripped_sigma_matrix = sigma_matrix[:k, :k]
        sum_k_sigma += np.trace(stripped_sigma_matrix)
    return (sum_k_sigma/sum_sigma)*100


img = cv2.imread(str(sys.argv[1]))
blue_channel = im2double(img[:, :, 0])
green_channel = im2double(img[:, :, 1])
red_channel = im2double(img[:, :, 2])

[u_red, s_red, vt_red] = channel_svd(red_channel)
[u_blue, s_blue, vt_blue] = channel_svd(blue_channel)
[u_green, s_green, vt_green] = channel_svd(green_channel)

rank_channel = np.linalg.matrix_rank(red_channel)

plot_singular_values(s_red, s_blue, s_green)

optimal_k_red = calculate_optimal_k(s_red, rank_channel)
optimal_k_blue = calculate_optimal_k(s_blue, rank_channel)
optimal_k_green = calculate_optimal_k(s_green, rank_channel)
optimal_k = int(np.median(np.asarray([optimal_k_red, optimal_k_blue, optimal_k_green])))

s_red_diagonalize = np.diag(s_red)
s_blue_diagonalize = np.diag(s_blue)
s_green_diagonalize = np.diag(s_green)
per_info = percentage_of_info_restored(optimal_k,
                                       [s_red_diagonalize, s_blue_diagonalize, s_green_diagonalize])

print "Number of dimensions in the image:", rank_channel
print "Number of dimensions used in reconstruction:", optimal_k
print "Percentage of information restored:", per_info

red_frobenius_norm = calculate_frobenius_norm(red_channel, u_red, s_red_diagonalize, vt_red)
blue_frobenius_norm = calculate_frobenius_norm(blue_channel, u_blue, s_blue_diagonalize, vt_blue)
green_frobenius_norm = calculate_frobenius_norm(green_channel, u_green, s_green_diagonalize, vt_green)

plot_frobenius_norm(red_frobenius_norm, blue_frobenius_norm, green_frobenius_norm, rank_channel)

print "Frobenius Norm of Red channel:", red_frobenius_norm[optimal_k]
print "Frobenius Norm of Blue channel:", blue_frobenius_norm[optimal_k]
print "Frobenius Norm of Green channel:", green_frobenius_norm[optimal_k]

blue_reconstruction_matrix = channel_via_optimal_k(optimal_k, u_blue, s_blue_diagonalize, vt_blue)
green_reconstruction_matrix = channel_via_optimal_k(optimal_k, u_green, s_green_diagonalize, vt_green)
red_reconstruction_matrix = channel_via_optimal_k(optimal_k, u_red, s_red_diagonalize, vt_red)

re_image = cv2.merge((blue_reconstruction_matrix, green_reconstruction_matrix, red_reconstruction_matrix))
cv2.imwrite('restored_image.png', re_image)
