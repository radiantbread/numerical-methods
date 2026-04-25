import numpy as np

def gauss_solve(A, b, pivot=None):
    """
    Решить систему Ax=b методом Гаусса.

    Параметры:
    A : массив numpy, матрица коэффициентов
    b : массив numpy, вектор
    pivot :
        "column" - используется выбор ведущего элемента по столбцу
        "full" - используется выбор ведущего элемента по всей матрице
        иначе - простой метод Гаусса

    Возвращает:
    x : массив numpy, вектор
    """
    # Скопируем матрицы, чтобы не изменять изначальные.
    A = A.copy()
    b = b.copy()
    dim = len(b)

    # При выборе ведущего элемента по матрице необходимо следить за порядком столбцов,
    # так как это влияет на правильность порядка компонент вектора.
    if pivot == "full":
        col_order = np.arange(dim) # Для n = 6 это вектор [0 1 2 3 4 5]
    
    for j in range(dim-1):
        if pivot == "column":
            row_max = j + np.argmax(np.abs(A[j:, j]))

            # При необходимости меняем столбцы местами.
            if row_max != j:
                A[[j, row_max]] = A[[row_max, j]]
                b[[j, row_max]] = b[[row_max, j]]
        elif pivot == "full":
            A_sub = A[j:, j:]
            elem_max = np.unravel_index(np.argmax(np.abs(A_sub)), A_sub.shape)
            row_max = j + elem_max[0]
            col_max = j + elem_max[1]
            
            if row_max != j:
                A[[j, row_max]] = A[[row_max, j]]
                b[[j, row_max]] = b[[row_max, j]]
            if col_max != j:
                A[:, [j, col_max]] = A[:, [col_max, j]]
                col_order[[j, col_max]] = col_order[[col_max, j]] # меняем порядок.
            
        # Преобразуем матрицу к верхнетреугольной при помощи элементарных преобразований строк.
        for i in range(j+1, dim):
            factor = A[i, j] / A[j, j]
            A[i, j:] -= factor * A[j, j:]
            b[i] -= factor * b[j]
    
    # Находим x из преобразованной матрицы "снизу вверх".
    x = np.zeros(dim)
    for i in range(dim - 1, -1, -1):
        x[i] = (b[i] - (A[i, i+1:] @ x[i+1:])) / A[i, i]
        
    if pivot == "full":
        # Возвращаем элементы x на место соответствующей перестановкой.
        x_unscrambled = np.zeros(dim)
        x_unscrambled[col_order] = x
        return x_unscrambled
    return x

def norm(A, order=1):
    """
    Получить норму матрицы соответствующего порядка

    Параметры:
    A : массив numpy, матрица
    order : число, порядок нормы
        np.inf - бесконечный порядок
        иначе 1
    
    Возвращает норму
    """
    dim = A.shape[0]
    ones = np.ones(dim)
    
    if order == np.inf:
        return np.max(np.abs(A)@ones)
    else:
        return np.max(ones@np.abs(A))

def cond(A, order=1):
    """
    Получить число согласованности матрицы соответствующего порядка

    Параметры:
    A : массив numpy, матрица
    order : число, порядок нормы
        np.inf - бесконечный порядок
        иначе 1
    
    Возвращает число согласованности
    """
    A_inv = gauss_inverse(A, pivot="column")
    return norm(A, order) * norm(A_inv, order)

def gauss_inverse(A, pivot=None):
    """
    Найти обратную матрицу, решая подсистемы AX=E.

    Параметры:
    A : массив numpy, матрица.
    pivot : для метода Гаусса
        "column" - используется выбор ведущего элемента по столбцу
        "full" - используется выбор ведущего элемента по всей матрице
        иначе - простой метод Гаусса
    
    Возвращает:
    inverse : массив numpy, матрица, обратная A
    """
    dim = A.shape[0]
    inverse = np.zeros((dim, dim))

    for i in range(dim):
        # Генерируем единичный вектор
        e = np.zeros(dim)
        e[i] = 1.0

        # Решить Ax = e_i
        inverse[:, i] = gauss_solve(A, e, pivot)
    
    return inverse