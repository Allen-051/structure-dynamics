import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import LogLocator, FuncFormatter
# 1. 讀取地震資料
file_path = input('Please enter an earthquake history data path (e.g., El Centro.txt): ')

try:
    # 2. 自動檢測檔案內容
    with open(file_path, 'r') as f:
        # 讀取檔案的標題行
        headers = f.readline().strip().split()
        print("Detected headers:", headers)

        # 判斷是否包含所需的關鍵字
        time_index = headers.index("Time(s)")  # 偵測關鍵字
        acc_index = headers.index("Acc.(g)")  # 偵測關鍵字

    # 3. 使用正確的分隔符號讀取資料
    data = np.genfromtxt(file_path, skip_header=1)

    # 4. 儲存關鍵欄位數值
    time_column = data[:, time_index]
    acc_column = data[:, acc_index]
    print("time：\n"+", ".join(map(str, time_column)))
    print("earthquake acc\n："+", ".join(map(str, acc_column)))
except FileNotFoundError:
    print("Error: The file was not found. Please check the path.")
except ValueError:
    print("Error: Missing required keywords ('Time', 'Acc') in file headers.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# 2. 設定週期範圍與自然頻率
Tn = np.arange(0.05, 10.00, 0.05)  # 週期範圍 0.05 ~ 10.0s
omega_n = 2 * np.pi / Tn  # 自然頻率範圍
m = float(input("Please enter the mass (unit：kg) of the system: "))
zeta = float(input("Please enter the damping ratio (zeta) of the system: "))
k = float(input("Please enter the stiffness (unit：N/m) of the system: "))
u0 = float(input("Please enter the initial displacement (unit：m) of the system: "))
v0 = float(input("Please enter the initial moving velocity (unit：m/s) of the system: "))

# 3 數值方法函數
def central_difference_method(m, c, k, p, time, u0, v0):
    dt = time[1] - time[0]
    n = len(time)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    u[0] = u0
    v[0] = v0
    a[0] = (p[0] - c * v[0] - k * u[0]) / m
    k_eff = m / dt**2 + c / (2 * dt)
    u[-1] = u[0] - dt * v[0] + 0.5 * a[0] * dt ** 2
    A = m / dt**2 - c/ (2 * dt)
    B = k - 2 * m / dt**2
    for i in range(0, n-1):
        u[i+1] = (p[i] - A * u[i-1] - B * u[i]) / k_eff
        v[i] = (u[i+1] - u[i-1]) / (2 * dt)
        a[i] = (u[i+1]- 2 * u[i] + u[i-1]) / dt**2
    return u, v, a


def linear_acceleration_method(m, c, k, p, time, u0, v0):
    dt = time[1] - time[0]  # 時間步長
    n = len(time)  # 時間步數
    # 初始化位移、速度、加速度陣列
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # 初始條件
    u[0] = u0
    v[0] = v0
    a[0] = (p[0] - c * v[0] - k * u[0]) / m  # 計算初始加速度

    # Newmark 線性加速度法的參數
    gamma = 0.5  # Newmark 方法中的速度加權因子
    beta = 1 / 6  # Newmark 方法中的位移加權因子

    # 預計算常數
    k_eff = k + (gamma * c) / (beta * dt) + m / (beta * dt**2)
    A = m / (beta * dt) + c *( gamma / beta)
    B = m / (2 * beta) + dt *( gamma /(2 * beta) -1)

    # Newmark 迭代
    for i in range(0 , n-1):
        delta_p_head = (p[i+1] - p[i]) + A * v[i] + B * a[i]
        delta_u = delta_p_head/ k_eff
        delta_v = gamma* delta_u/(beta* dt) - gamma * v[i]/beta + dt * a[i] * (1 - gamma /(2 * beta))
        delta_a = delta_u/(beta* dt**2) - v[i]/(beta*dt) - a[i]/(2 *beta)
        u[i+1] = u[i] + delta_u
        v[i+1] = v[i] + delta_v
        a[i+1] = a[i] + delta_a

    return u, v, a
# 儲存最大反應

max_u1, max_v1, max_a1 = [], [], []
max_u2, max_v2, max_a2 = [], [], []

# 迭代計算不同自然頻率的反應
for omega in omega_n:
    k = m * omega ** 2
    c = 2 * zeta * omega * m

    u1, v1, a1 = central_difference_method(m, c, k, -m * acc_column, time_column, u0, v0)
    u2, v2, a2 = linear_acceleration_method(m, c, k, -m * acc_column, time_column, u0, v0)

    max_u1.append(np.max(np.abs(u1)))
    max_v1.append(np.max(np.abs(v1)))
    max_a1.append(np.max(np.abs(a1)))

    max_u2.append(np.max(np.abs(u2)))
    max_v2.append(np.max(np.abs(v2)))
    max_a2.append(np.max(np.abs(a2)))

# 4. 詢問使用者繪製哪種反應譜
choice = input("Enter '1' for Displacement Spectrum, '2' for Velocity Spectrum, or '3' for Acceleration Spectrum: ")
plt.figure(figsize=(10, 6))
if choice == '1':
    plt.plot(Tn, max_u1, 'green', label='Central Difference (u1)')
    plt.plot(Tn, max_u2, 'red', label='Linear Acceleration (u2)')
    plt.ylabel('Maximum Displacement (m)')
    plt.title('Displacement Response Spectrum')
elif choice == '2':
    plt.plot(Tn, max_v1, 'orange', label='Central Difference (v1)')
    plt.plot(Tn, max_v2, 'blue', label='Linear Acceleration (v2)')
    plt.ylabel('Maximum Velocity (m/s)')
    plt.title('Velocity Response Spectrum')
elif choice == '3':
    plt.plot(Tn, max_a1, 'purple', label='Central Difference (a1)')
    plt.plot(Tn, max_a2, 'pink', label='Linear Acceleration (a2)')
    plt.ylabel('Maximum Acceleration (m/s²)')
    plt.title('Acceleration Response Spectrum')
else:
    print("Invalid choice.")
    exit()

plt.xlabel('Period (Tn) (s)')
plt.xscale('log')
plt.yscale('log')
#plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=None))  # 僅顯示 10^0, 10^1 等
#plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):d}"if x.is_integer() else f"{x:g}"))
#plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs=None))
#plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):d}"if y.is_integer() else f"{y:g}"))
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.2)
plt.show()

# 5. 儲存圖片
save_path = input("Please tell me where to save the diagram.( e.g. 'C；/user/...')")

plt.savefig(save_path)
print(f"Response spectrum saved to {save_path}")
