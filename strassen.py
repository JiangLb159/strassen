import numpy as np
import math


# 将矩阵分为4块，返回其中的指定部分
def matrix_divide(matrix):
    Ax = np.split(matrix, 2)
    A1 = Ax[0]
    A2 = Ax[1]

    Axx = np.split(A1, 2, axis=1)
    A11 = Axx[0]
    A12 = Axx[1]
    Axx = np.split(A2, 2, axis=1)
    A21 = Axx[0]
    A22 = Axx[1]
    return A11, A12, A21, A22


# 普通strassen方法，必须为2^k维方阵
def strassen(matrix_a, matrix_b):
    rows = len(matrix_a)
    if rows == 1:
        matrix_res = np.array([[matrix_a[0][0] * matrix_b[0][0]]])
    else:
        A11, A12, A21, A22 = matrix_divide(matrix_a)
        B11, B12, B21, B22 = matrix_divide(matrix_b)

        s1 = B12 - B22
        s2 = A11 + A12
        s3 = A21 + A22
        s4 = B21 - B11
        s5 = A11 + A22
        s6 = B11 + B22
        s7 = A12 - A22
        s8 = B21 + B22
        s9 = A11 - A21
        s10 = B11 + B12
        p1 = strassen(A11, s1)
        p2 = strassen(s2, B22)
        p3 = strassen(s3, B11)
        p4 = strassen(A22, s4)
        p5 = strassen(s5, s6)
        p6 = strassen(s7, s8)
        p7 = strassen(s9, s10)

        c11 = p5 + p4 - p2 + p6
        c12 = p1 + p2
        c21 = p3 + p4
        c22 = p5 + p1 - p3 - p7
        # 左右合并
        c1 = np.concatenate((c11, c12), axis=1)
        c2 = np.concatenate((c21, c22), axis=1)
        # 上下合并
        matrix_res = np.concatenate((c1, c2), axis=0)

    return matrix_res


# 将方阵拆为m个2^k阶方阵，A为4维列表，2维位置，2维矩阵
def strassen_expand_matrix_divide(matrix, div):
    A = np.split(matrix, div, axis=0)  # [1,2,3]
    A = np.array([np.split(i, div, axis=1) for i in A])
    return A


# 使阶数n=m*2^k的方阵也可以使用strassen方法
def strassen_expand(matrix_a, matrix_b):
    length = len(matrix_a)  # 12
    # 判断是否为普通型
    simple = math.log(length)
    if simple == int(simple):  # 是普通型
        return strassen(matrix_a, matrix_b)

    # 不是普通型
    # div：分为div维矩阵，每个元素为一个strassen方阵
    div = length
    while True:
        if div / 2 == int(div / 2):
            div = div / 2
        else:
            break

    A = strassen_expand_matrix_divide(matrix_a, div)
    B = strassen_expand_matrix_divide(matrix_b, div)

    return matrix_multiplication(A, B, div)


# 4维矩阵乘法
def matrix_multiplication(matrix_a, matrix_b, div):
    length = len(matrix_a[0][0])
    matrix_res = np.zeros(shape=matrix_a.shape)

    for i in range(int(div)):
        for j in range(int(div)):
            # 空矩阵，用于储存矩阵的和
            tmp = np.zeros((length, length))
            for k in range(int(div)):
                tmp = tmp + strassen(matrix_a[i][k], matrix_b[k][j])
            matrix_res[i][j] = tmp

    # 聚合，使列合为一起，即左右合并
    cons = []
    for i in range(int(div)):
        con = matrix_res[i][0]
        for j in range(1, int(div)):
            con = np.concatenate((con, matrix_res[i][j]), axis=1)
        cons.append(con)

    # 聚合，上下合并
    matrix_res = cons[0]
    for i in range(1, len(cons)):
        matrix_res = np.concatenate((matrix_res, cons[i]), axis=0)

    return matrix_res


def main():
    a = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    b = np.array([[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]])
    c = strassen(a, b)
    print(c)
    d = a @ b
    print(d == c)
    print(d)

    print("=============================================")
    # 使阶数n=m*2^k的方阵也可以使用strassen方法
    # 12阶方阵相乘，3*4=3*(2^2)，例子中div=3
    a = np.arange(1, 145).reshape((12, 12))
    print(type(a))
    print(a)
    b = np.eye(12, 12)
    print(b)
    c = strassen_expand(a, b)
    print(c)
    d = a @ b
    print(d == c)
    print(d)


if __name__ == '__main__':
    main()
