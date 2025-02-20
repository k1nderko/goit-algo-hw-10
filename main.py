import pulp
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# Завдання 1: Оптимізація виробництва
# Оголошуємо задачу лінійного програмування
model = pulp.LpProblem("Maximize_Production", pulp.LpMaximize)

# Змінні: кількість одиниць виробленого Лимонаду та Фруктового соку
lemonade = pulp.LpVariable("Lemonade", lowBound=0, cat='Integer')
fruit_juice = pulp.LpVariable("Fruit_Juice", lowBound=0, cat='Integer')

# Функція цілі: максимізація загальної кількості продуктів
model += lemonade + fruit_juice

# Обмеження ресурсів
model += 2 * lemonade + 1 * fruit_juice <= 100  # Вода
model += 1 * lemonade <= 50  # Цукор
model += 1 * lemonade <= 30  # Лимонний сік
model += 2 * fruit_juice <= 40  # Фруктове пюре

# Розв’язання задачі
model.solve()

# Виведення результатів
print("Завдання 1: Оптимізація виробництва")
print("Лимонад:", lemonade.varValue)
print("Фруктовий сік:", fruit_juice.varValue)
print("Максимальна загальна кількість продукції:", lemonade.varValue + fruit_juice.varValue)

# Завдання 2: Обчислення визначеного інтеграла методом Монте-Карло

def f(x):
    return x ** 2

a, b = 0, 2  # Межі інтегрування
num_points = 100000  # Кількість випадкових точок

# Генеруємо випадкові точки
x_rand = np.random.uniform(a, b, num_points)
y_rand = np.random.uniform(0, max(f(x_rand)), num_points)

# Обчислення площі під кривою методом Монте-Карло
under_curve = y_rand <= f(x_rand)
area_mc = (sum(under_curve) / num_points) * (b - a) * max(f(x_rand))

# Аналітичне обчислення інтеграла
result, error = spi.quad(f, a, b)

# Виведення результатів
print("\nЗавдання 2: Обчислення визначеного інтеграла")
print("Метод Монте-Карло:", area_mc)
print("Аналітичне обчислення:", result)
print("Похибка Монте-Карло:", abs(area_mc - result))

# Візуалізація
x = np.linspace(a - 0.5, b + 0.5, 400)
y = f(x)
fig, ax = plt.subplots()
ax.plot(x, y, 'r', linewidth=2)
ix = np.linspace(a, b)
iy = f(ix)
ax.fill_between(ix, iy, color='gray', alpha=0.3)
ax.scatter(x_rand[:1000], y_rand[:1000], c=under_curve[:1000], cmap='coolwarm', alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title(f'Графік інтегрування f(x) = x^2 від {a} до {b}')
plt.grid()
plt.show()
