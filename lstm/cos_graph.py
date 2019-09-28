import numpy as np
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        pass

    def show(self):
        signal_data = np.cos(np.arange(1600) * (20 * np.pi / 1000))[:, None]
        plot_x = np.arange(1600)
        plot_y = signal_data
        plt.plot(plot_x, plot_y)
        plt.show()

"""
데이터 셋을 코사인 데이터로 생성함.
시간의 흐름에 따라 진폭이 -1.0에서 1.0사이로 변하는 1600개의 실수값을 생성.
지금의 값을 바탕으로 동일한 진폭을 유지한다고 했을때 미래의 어느시점에서 어떤값일지를 예측.
"""

if __name__ == '__main__':
    g = Graph()
    g.show()

