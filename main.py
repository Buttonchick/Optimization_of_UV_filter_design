import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.stats import norm, lognorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad


m_mu = 3.2 # математическое ожидание массы частицы
m_sigma = 1.6  # стандартное отклонение массы частицы

# Начальное распределение зарядов
q_mu =  -88.43427374661621
q_sigma =  132.59252696990796
q_A = 0.1314373943594196

# задаем начальные условия
e = 1.6e-19 # Кл

v3=0.001 # м^3/c расход воздуха при дыхании среднего человека 
t = 0.1 # мм, оптимальная толщина электрода по ГОСТ 24045-2019
U = 5 # В, исходя из характеристик диодов 
dt = 1e-4 # с
K = 0.9 # кофициент электрода для  меди 



Ds = [40,60]  # мм, стандартные диаметры по ГОСТ Р 12.4.191-99
ls = [ 4, 5, 6.3, 8, 10, 12.5, 16, 20]  # мм, стандартные длины по ГОСТ 427-75
ns = range ( 2 , 40 + 2 , 2 )

ms = np.logspace(-4*m_sigma, 4*m_sigma, 1000)  # нг, вероятные значения массы
qs = np.round(np.linspace(q_mu - 3 * q_sigma, q_mu + 3 * q_sigma, int(6 * abs(q_sigma))))  # е, вероятные значения зарядов



electrodes = np.array([(D_max, L, n) for D_max in Ds for L in ls for n in ns])
aerosols = np.array([(m, q) for m in ms for q in qs ])





def probability(m, q):
    """
    Рассчитывает вероятность события на основе логнормального распределения массы частицы и гауссовского распределения заряда.

    Аргументы:
    m -- масса частицы
    q-- заряда частицы

    Возвращает:
    Вероятность события
      """
    # расчет вероятности события 
    prob = lognorm.pdf(m, m_sigma, scale=np.exp(m_mu)) * q_A* np.exp(-(q-q_mu)**2/(2*q_sigma**2))
        
    return np.clip(100*prob, 0.03, 99.97)



def particle_t(q, V, m, x0, v):
	  
  	# вычисляем константы
    k = 1/(4 * math.pi * 8.85e-12) 
    c = np.sqrt(abs((2*m)/(-q*V*k)))
  
    t = x0*c # вычисляем время t
    z = v*t # вычисляем глубину оседания
      
    return t, z
  

def particle_motion(q, V, m, d, x0, v0, theta0, dt):
    """
    Рассчитывает траекторию движения заряженной частицы в электрическом поле между двумя концентрическими соосными электродами
    с радиусами R1 и R2, на которые подано напряжение U, в плоскости xz.
    Возвращает массив координат (x, z) и скоростей (v, theta) на каждом временном шаге до момента оседания частицы на электроде.
    
    Параметры:
    q - заряд частицы (Кл)
    m - масса частицы (кг)
    V - потенциал поля (В) 
    d - расстояние между электроодами (м)
    x0 - начальная координата частицы по оси x (м)
    v0 - начальная скорость частицы (м/с)
    theta0 - начальный угол между осью x и направлением движения частицы (рад)
    dt - временной шаг (с)
    """
    
    # определение констант
    k = 1/(4 * math.pi * 8.85e-12)
    
  
    # начальные условия
    x, z = x0, 0
    v, theta = v0, theta0
    vz = v0 * math.cos(theta) 
    vx = v0 * math.sin(theta)
    
    # массивы координат и скоростей
    positions = [(x, z)]
    velocities = [(v, theta)]
    
    # вычисление траектории
    while 0 < x < d:
      
     
        # вычисление напряженности поля в текущей точке
        E = V/x*k
       
        # вычисление ускорения частицы
        a = -q/m * E
        
        # вычисление новой скорости и угла
        vz = v0
        vx += a*dt
        
        # вычисление новых координат
        x += vx * dt
        z += vz * dt
        if x < 0:
          x = 0
        if x > d:
          x = d

        
        # добавление координат и скоростей в массивы
        positions.append((x, z))
        velocities.append((vx, vz))
        
    return positions, velocities


                

def pressure_drop(v, L, D_eff, rho=1.2, mu=1.8e-5):
    # Расчет числа Рейнольдса
    Re = (rho * v * D_eff) / mu

    k = 1.5e-6
    
    # Расчет коэффициента трения
    f = (0.25) / (math.log10((k / (3.7 * D_eff)) + (5.74 / (Re ** 0.9))) ** 2)
    
    # Потери давления в фильтре
    delta = f *(L/ D_eff)* (rho * v**2 / 2)
    
    return delta



electrode_params = []

for electrode in electrodes:
    D_max, L, n = electrode  # Распаковка значений электрода

    # Радиус внешнего электрода
    R_outer = 0.001 * D_max / 2

    # Радиус внутреннего электрода
    R_inner = R_outer / n

    # Расстояние между электродами
    d = (R_outer - R_inner) / (n - 1)

    # Эффективный диаметр фильтра
    R_eff = R_outer - R_inner - 0.001 * t * (n - 1)

    # Площадь поперечного сечения трубы
    A = math.pi * R_eff ** 2

    # Средняя скорость потока
    if Ds.index(D_max) == 0:
        v = 2 * v3 / A  # скорость при D = 40 мм, двойной фильтр
    elif Ds.index(D_max) == 1:
        v = v3 / A  # скорость при D = 60 мм, одиночный фильтр
    else:
        v = v3 / A  # скорость

    # Электрическая постоянная поля
    V = U * math.log(R_outer / R_inner) / (n / 2)

    # Спротивление вдоху
    delta = pressure_drop(v, 0.001 * L, 2 * R_eff)

    z_max = 0

    lost_aerosols = []
    cought_aerosols = []

    # Итерация аэрозолей
    aerosols = np.array(aerosols)
    m = aerosols[:, 0]
    q = aerosols[:, 1]

    _, z = particle_t(q * e, V, m * 1e-12, d / 2, v)
    probs = probability(m, q)

    cought_mask = z <= 0.001 * L / 2
    cought_aerosols = aerosols[cought_mask]
    lost_aerosols = aerosols[~cought_mask]

    cought_prob = np.sum(probs[cought_mask])
    lost_prob = np.sum(probs[~cought_mask])
    eff = cought_prob / (cought_prob + lost_prob)

    electrode_params.append((electrode, delta, eff*100))





# Создайте списки для хранения значений L, eff и n
L_values = []
eff_values = []
n_values = []
delta_values = []

# Итерация по значениям в electrode_params
for params in electrode_params:
    electrode, delta, eff = params  # Распаковка значений
    D_max, L, n = electrode  # Распаковка значений электрода

    # Добавляем значения в соответствующие списки
    L_values.append(L)
    eff_values.append(eff)
    n_values.append(n)
    delta_values.append(delta)

print("для D = 40мм")
for k in [0.683,0.954,0.997]:
  eff_threshold = max([params[2] for params in electrode_params])  # заданная эффективность
  filtered_D40_params = [params for params in electrode_params if params[2] >= round(k*eff_threshold,3) and params[0][0] == 40]
  
  if len(filtered_D40_params) == 0: # проверяем наличие подходящих значений
        continue # если список пустой, продолжаем цикл
    
  min_L = min([params[0][1] for params in filtered_D40_params])
  min_delta = min([params[1] for params in filtered_D40_params])
  opt_n = [params[0][2] for params in filtered_D40_params if params[1] == min_delta]

  print(f"для эффективности {round(k*eff_threshold, 3)}%")
  print(f"Оптимальные значения ширины L = {min_L}, мм, количества электродов n = {opt_n}, шт для достижения, сопротивлении вдоху delt = {round(min_delta*1000,1)}, мПа")



print("для D = 60мм")
for k in [0.683,0.954,0.997,0.9994]:
  filtered_D60_params = [params for params in electrode_params if params[2] >= round(k*eff_threshold, 3) and params[0][0] == 60]
  
  if len(filtered_D60_params) == 0: # проверяем наличие подходящих значений
          continue # если список пустой, продолжаем цикл
        
  min_L = min([params[0][1] for params in filtered_D60_params])
  min_delta = min([params[1] for params in filtered_D60_params])
  opt_n = [params[0][2] for params in filtered_D60_params if params[1] == min_delta]

  print(f"для эффективности {round(k*eff_threshold, 3)}%")
  
  print(f"Оптимальные значения ширины L = {min_L}, мм, количества электродов n = {opt_n}, шт для достижения, сопротивлении вдоху delt = {round(min_delta*1000,1)}, мПа")
 
  



# Построение графиков зависимости эффективности от n для разных L
plt.figure()


# График для D = 40 мм
plt.subplot(1, 2, 1)
for L_value in set(L_values):
    eff_for_L = [eff_values[i] for i in range(len(L_values)) if L_values[i] == L_value and Ds.index(electrode_params[i][0][0]) == 0]
    n_for_L = [n_values[i] for i in range(len(L_values)) if L_values[i] == L_value and Ds.index(electrode_params[i][0][0]) == 0]
    plt.plot(n_for_L, eff_for_L, label=f'Ширина L = {L_value}, мм')
plt.xlabel('Количество электродов n')
plt.ylabel('Эффективность, %')
plt.title('D = 40 mm')
plt.legend()

# График для D = 60 мм
plt.subplot(1, 2, 2)
for L_value in set(L_values):
    eff_for_L = [eff_values[i] for i in range(len(L_values)) if L_values[i] == L_value and Ds.index(electrode_params[i][0][0]) == 1]
    n_for_L = [n_values[i] for i in range(len(L_values)) if L_values[i] == L_value and Ds.index(electrode_params[i][0][0]) == 1]
    plt.plot(n_for_L, eff_for_L, label=f'Ширина L = {L_value}, мм')
plt.xlabel('Количество электродов n')
plt.ylabel('Эффективность, %')
plt.title('D = 60 mm')
plt.legend()

plt.tight_layout()
plt.show()

      
# Построение графиков зависимости delta от n для разных L
plt.figure()

# График для D = 40 мм
plt.subplot(1, 2, 1)
for L_value in set(L_values):
    delta_for_L = [electrode_params[i][1] for i in range(len(L_values)) if L_values[i] == L_value and Ds.index(electrode_params[i][0][0]) == 0]
    n_for_L = [n_values[i] for i in range(len(L_values)) if L_values[i] == L_value and Ds.index(electrode_params[i][0][0]) == 0]
    plt.plot(n_for_L, delta_for_L, label=f'Ширина L = {L_value}, мм')
plt.xlabel('Количество электродов n')
plt.ylabel('Сопротивление вдоху, Па')
plt.title('D = 40 mm')
plt.legend()

# График для D = 60 мм
plt.subplot(1, 2, 2)
for L_value in set(L_values):
    delta_for_L = [electrode_params[i][1] for i in range(len(L_values)) if L_values[i] == L_value and Ds.index(electrode_params[i][0][0]) == 1]
    n_for_L = [n_values[i] for i in range(len(L_values)) if L_values[i] == L_value and Ds.index(electrode_params[i][0][0]) == 1]
    plt.plot(n_for_L, delta_for_L, label=f'Ширина L = {L_value}, мм')
plt.xlabel('Количество электродов n')
plt.ylabel('Сопротивление вдоху, Па')
plt.title('D = 60 mm')
plt.legend()

plt.tight_layout()
plt.show()