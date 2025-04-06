import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
text_scores = None
text_mean = None
text_recent = None

def plot(scores, mean_scores, mean_recent_scores):
    global text_scores, text_mean, text_recent

    ax.clear()
    ax.set_title('Training...')
    ax.set_xlabel('Number of games')
    ax.set_ylabel('Score')
    ax.plot(scores, label='Score')
    ax.plot(mean_scores, label='Mean Score')
    ax.plot(mean_recent_scores, label='Recent Mean Score')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')

    if len(scores) > 0:
        ax.text(len(scores)-1, scores[-1], str(scores[-1]))
        ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        ax.text(len(mean_recent_scores)-1, mean_recent_scores[-1], str(mean_recent_scores[-1]))
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)