import matplotlib.pyplot as plt
import imageio
import numpy as np

def saveTrainingPlot(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color='blue', linewidth=1)
    ax.set_xlabel("Game", color='blue')
    ax.set_ylabel("Epsilon", color='blue')
    ax.tick_params(axis='x', colors='blue')
    ax.tick_params(axis='y', colors='blue')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color='green', linewidth=1)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='green')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='green')

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def saveTestingPlot(x, scores, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    ax.plot(x, scores, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("scores", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def saveTrainingGif(images):
    imageio.mimsave('LunarLander.gif',
                    [np.array(img) for i, img in enumerate(images) if
                     i % 2 == 0], fps=29)


def saveHist(test_score, filename):
    # Гистограмма
    a = len([x for x in test_score if x < 0])
    b = len([x for x in test_score if 0 <= x < 100])
    c = len([x for x in test_score if 100 <= x < 200])
    d = len([x for x in test_score if x > 200])

    x = ['0-', '0+', '100+', '200+']
    y = [a, b, c, d]

    fig, ax = plt.subplots()

    ax.bar(x, y)

    ax.set_facecolor('seashell')
    fig.set_facecolor('floralwhite')
    fig.set_figwidth(6)
    fig.set_figheight(6)

    for i, v in enumerate(y):
        ax.text(i, v + 3, str(v), color='blue', fontweight='bold')

    # plt.show()
    plt.savefig(filename)
