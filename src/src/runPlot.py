import numpy as np
import matplotlib.pyplot as plt
import pickle
if __name__ == '__main__':

    with open("T1_dist_300.txt", "r") as file:
        d_t1 = np.array(eval(file.readline()))
    with open("T2_dist_300.txt", "r") as file:
        d_t2 = np.array(eval(file.readline()))
    with open("T3_dist_300.txt", "r") as file:
        d_t3 = np.array(eval(file.readline()))

    d_sum = (d_t1 + d_t2 + d_t3)
    d_sum = d_sum / 3

    with open("T1_heading_300.txt", "r") as file:
        h_t1 = np.array(eval(file.readline()))
    with open("T2_heading_300.txt", "r") as file:
        h_t2 = np.array(eval(file.readline()))
    with open("T3_heading_300.txt", "r") as file:
        h_t3 = np.array(eval(file.readline()))

    h_sum = (h_t1 + h_t2 + h_t3)
    h_sum = h_sum / 3


    fig, axs = plt.subplots(2, 1)
    axs[0].plot(d_sum)
    #axs[0].set_title('subplot 1')
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('distance error')
    fig.suptitle('3 Runs - 300 Particles Average Error', fontsize=16)

    axs[1].plot(h_sum)
    axs[1].set_xlabel('iterations')
    #axs[1].set_title('subplot 2')
    axs[1].set_ylabel('heading error')

    plt.show()

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(d_sum)
    # plt.xlabel('Iterations')
    # plt.ylabel('Distance Error')
    #
    # plt.subplot(212)
    # plt.plot(h_sum)
    # plt.xlabel('Iterations')
    # plt.ylabel('Heading Error')
    # fig.suptitle('300 Particles Average Error', fontsize=10)
    # plt.show()
