import math
import numpy as np
from scipy.optimize import bisect, newton, root_scalar

print("Решается уравнение: f(x) = x^3 + 2x - 5 = 0")
print("Производная: f'(x) = 3x^2 + 2")
print("Функция для простой итерации: phi(x) = (5 - 2x)^(1/3)")
print("Интервал: [1; 2]")
print()

def f(x):
    return x ** 3 + 2 * x - 5


def df(x):
    return 3 * x ** 2 + 2


def phi(x):
    v = 5 - 2 * x
    return math.copysign(abs(v) ** (1 / 3), v)


def tabulate_and_locate(a, b, step):
    print("Табулирование функции на отрезке [%s; %s] с шагом %s" % (a, b, step))
    print("%-6s %-12s" % ("x", "f(x)"))
    grid = []
    n = int(round((b - a) / step)) + 1
    for i in range(n):
        x = round(a + i * step, 10)
        grid.append(x)
        print("%-6.2f %-12.6f" % (x, f(x)))
    print()
    for i in range(len(grid) - 1):
        if f(grid[i]) * f(grid[i + 1]) < 0:
            print("Корень локализован на отрезке [%s; %s]" % (grid[i], grid[i + 1]))
            return grid[i], grid[i + 1]
    return None, None


def bisection(a, b, eps):
    iters = 0
    prev = a
    print("\nМетод бисекции")
    print("Точность: %.0e" % eps)
    print("%-4s %-12s %-12s %-12s %-12s" % ("N", "x_prev", "x_new", "error", "f(x_new)"))
    while True:
        iters += 1
        x_new = (a + b) / 2
        error = abs(x_new - prev)
        print("%-4d %-12.7f %-12.7f %-12.7f %-12.7f" % (iters, prev, x_new, error, f(x_new)))
        if error < eps:
            break
        if f(a) * f(x_new) < 0:
            b = x_new
        else:
            a = x_new
        prev = x_new
    residual = abs(f(x_new))
    print("Корень: %.7f" % x_new)
    print("Итераций: %d" % iters)
    print("Невязка: %.7f" % residual)
    return x_new, iters, residual


def chords(a, b, eps):
    iters = 0
    prev = a
    print("\nМетод хорд")
    print("Точность: %.0e" % eps)
    print("%-4s %-12s %-12s %-12s %-12s" % ("N", "x_prev", "x_new", "error", "f(x_new)"))
    while True:
        iters += 1
        x_new = a - f(a) * (b - a) / (f(b) - f(a))
        error = abs(x_new - prev)
        print("%-4d %-12.7f %-12.7f %-12.7f %-12.7f" % (iters, prev, x_new, error, f(x_new)))
        if error < eps:
            break
        if f(a) * f(x_new) < 0:
            b = x_new
        else:
            a = x_new
        prev = x_new
    residual = abs(f(x_new))
    print("Корень: %.7f" % x_new)
    print("Итераций: %d" % iters)
    print("Невязка: %.7f" % residual)
    return x_new, iters, residual


def newton_method(x0, eps):
    iters = 0
    x = x0
    print("\nМетод Ньютона")
    print("Начальное приближение: %s" % x)
    print("Точность: %.0e" % eps)
    print("%-4s %-12s %-12s %-12s %-12s" % ("N", "x", "x_new", "error", "f(x_new)"))
    while True:
        iters += 1
        x_new = x - f(x) / df(x)
        error = abs(x_new - x)
        print("%-4d %-12.7f %-12.7f %-12.7f %-12.7f" % (iters, x, x_new, error, f(x_new)))
        if error < eps:
            break
        x = x_new
    residual = abs(f(x_new))
    print("Корень: %.7f" % x_new)
    print("Итераций: %d" % iters)
    print("Невязка: %.7f" % residual)
    return x_new, iters, residual


def simple_iteration(x0, eps):
    iters = 0
    x = x0
    print("\nМетод простой итерации")
    print("Начальное приближение: %s" % x)
    print("Точность: %.0e" % eps)
    print("%-4s %-12s %-12s %-12s %-12s" % ("N", "x", "x_new", "error", "f(x_new)"))
    while True:
        iters += 1
        x_new = phi(x)
        error = abs(x_new - x)
        print("%-4d %-12.7f %-12.7f %-12.7f %-12.7f" % (iters, x, x_new, error, f(x_new)))
        if error < eps:
            break
        x = x_new
    residual = abs(f(x_new))
    print("Корень: %.7f" % x_new)
    print("Итераций: %d" % iters)
    print("Невязка: %.7f" % residual)
    return x_new, iters, residual


def check_convergence(x_check):
    phi_der = abs(-2 / (3 * (5 - 2 * x_check) ** (2 / 3)))
    print("\nПроверка сходимости простой итерации")
    print("|phi'(x)| = %.4f" % phi_der)
    if phi_der < 1:
        print("Условие сходимости выполняется")
    else:
        print("Условие сходимости НЕ выполняется")


def scipy_check():
    print("\nБиблиотечные методы SciPy")
    root_bisect = bisect(f, 1, 2, xtol=1e-10)
    print("SciPy bisect: %.10f" % root_bisect)
    root_newton = newton(f, 1.5, fprime=df, tol=1e-10)
    print("SciPy newton: %.10f" % root_newton)
    res = root_scalar(f, bracket=[1, 2], method='brentq', xtol=1e-10)
    print("SciPy brentq: %.10f" % res.root)
    return root_bisect


def print_summary_table(rows):
    print("\nИтоговая таблица")
    print("%-28s %-10s %-14s %-8s %-14s %-14s" % (
        "Метод", "eps", "Корень", "Итер.", "Невязка", "SciPy"
    ))
    for r in rows:
        print("%-28s %-10.0e %-14.8f %-8d %-14.8f %-14.8f" % (
            r[0], r[1], r[2], r[3], r[4], r[5]
        ))


def main():
    a, b = tabulate_and_locate(1.0, 2.0, 0.1)
    if a is None:
        print("Интервал с корнем не найден")
        return

    check_convergence(1.33)
    scipy_root = scipy_check()

    rows = []

    for eps in (1e-3, 1e-5):
        root, iters, res = bisection(a, b, eps)
        rows.append(("Бисекция", eps, root, iters, res, scipy_root))

        root, iters, res = chords(a, b, eps)
        rows.append(("Хорды", eps, root, iters, res, scipy_root))

        root, iters, res = newton_method(1.5, eps)
        rows.append(("Ньютон x0=1.5", eps, root, iters, res, scipy_root))

        root, iters, res = newton_method(1.1, eps)
        rows.append(("Ньютон x0=1.1", eps, root, iters, res, scipy_root))

        root, iters, res = simple_iteration(1.5, eps)
        rows.append(("Простая итерация x0=1.5", eps, root, iters, res, scipy_root))

        root, iters, res = simple_iteration(1.1, eps)
        rows.append(("Простая итерация x0=1.1", eps, root, iters, res, scipy_root))

    print_summary_table(rows)


if __name__ == "__main__":
    main()