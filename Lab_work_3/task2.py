"""
Завдання:
    Мінімізувати функцію Q = -2*X1 - 2*X2 за умов:
        1.5 * X1 + 2 * X2 - X4 <= 12
        X1 + 2 * X2 - X3 <=
        4 * X1 - X5 <= 16
        4 * X2 - X6 <= 12
        Xj >= 0, (j = 1..6)
"""

from ortools.linear_solver import pywraplp

# Створення лінійного розв'язувача
solver = pywraplp.Solver.CreateSolver('CP_SAT')

# Задаємо змінні X1, ..., X6 з умовою, що вони не можуть бути меншими за 0 і
# не більшими за 10 (для знаходження розв'язку в обмеженій області)
X1 = solver.NumVar(0, 10, 'X1')
X2 = solver.NumVar(0, 10, 'X2')
X3 = solver.NumVar(0, 10, 'X3')
X4 = solver.NumVar(0, 10, 'X4')
X5 = solver.NumVar(0, 10, 'X5')
X6 = solver.NumVar(0, 10, 'X6')

# Задаємо цільову функцію
solver.Minimize(-2 * X1 - 2 * X2)

# Задаємо обмеження
solver.Add(1.5 * X1 + 2 * X2 - X4 <= 12)
solver.Add(X1 + 2 * X2 - X3 <= 8)
solver.Add(4 * X1 - X5 <= 16)
solver.Add(4 * X2 - X6 <= 12)

# Виконуємо обчислення
status = solver.Solve()

# Виведення результатів
if status == pywraplp.Solver.OPTIMAL:
    print('Оптимальне рішення знайдено:')
    print(f'X1 = {X1.solution_value()}')
    print(f'X2 = {X2.solution_value()}')
    print(f'X3 = {X3.solution_value()}')
    print(f'X4 = {X4.solution_value()}')
    print(f'X5 = {X5.solution_value()}')
    print(f'X6 = {X6.solution_value()}')
    print(f'Мінімальне значення цільової функції = {-2 * X1.solution_value() - 2 * X2.solution_value()}')
else:
    print('Оптимальне рішення не знайдено.')
