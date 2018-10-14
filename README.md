# SVD based Image Compression

Singular value decomposition (SVD) is a linear algebra technique where a matrix is factored into product of three matrices, that is A = UΣV<sup>T</sup>. Σ is a diagonal matrix and its entries are called singular values. Interestingly for an image, only the top few singular values contains most of the "information" to represent the image. For further information refer: https://en.wikipedia.org/wiki/Singular_value_decomposition<br />
<br />
One application of SVD is data compression. Given a data matrix A (for instance an image), SVD can help to find a low rank matrix which is a good approximation of the original data matrix.<br />
<br />
In this project I have created a very basic image compression algorithm. Given a colored image, the algorithm will calculate the singular value decomposition of the image matrix. Then it will find the optimal number of dimensions required to get best tradeoff between reconstruction error and image fidelity. As a quality measure, I have used Frobenius Norm to calculate the reconstruction error.
