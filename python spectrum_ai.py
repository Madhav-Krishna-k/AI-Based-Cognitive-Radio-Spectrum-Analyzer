import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk

fs = 1000
t = np.arange(0, 1, 1/fs)

def generate_signal():
    s1 = np.sin(2*np.pi*50*t)
    s2 = np.sin(2*np.pi*120*t)
    s3 = np.sin(2*np.pi*200*t)
    s4 = np.sin(2*np.pi*260*t)

    signal = s1 + s2 + s3 + s4
    noise = np.random.normal(0, 0.5, len(t))

    return signal + noise


def spectrum(signal):
    N = len(signal)

    yf = fft(signal)
    xf = fftfreq(N, 1/fs)

    power = np.abs(yf)

    return xf, power


def classify(power):
    data = power.reshape(-1, 1)

    model = KMeans(n_clusters=2, n_init=10)

    model.fit(data)

    return model.labels_


def choose_channel(freq, power):
    threshold = np.mean(power)

    free = freq[power < threshold]

    if len(free) > 0:
        return free[0]

    return None


def stats(power):
    avg = np.mean(power)
    peak = np.max(power)

    return avg, peak


root = tk.Tk()
root.title("AI Cognitive Radio Spectrum Analyzer")

frame = ttk.Frame(root, padding=10)
frame.pack()

avg_label = ttk.Label(frame, text="Average Power:")
avg_label.pack()

peak_label = ttk.Label(frame, text="Peak Power:")
peak_label.pack()

channel_label = ttk.Label(frame, text="Recommended Channel:")
channel_label.pack()


fig, ax = plt.subplots()

def update(frame):

    signal = generate_signal()

    freq, power = spectrum(signal)

    mask = freq > 0
    freq = freq[mask]
    power = power[mask]

    classify(power)

    channel = choose_channel(freq, power)

    avg, peak = stats(power)

    avg_label.config(text=f"Average Power: {round(avg,2)}")
    peak_label.config(text=f"Peak Power: {round(peak,2)}")

    if channel is not None:
        channel_label.config(text=f"Recommended Channel: {round(channel,2)} Hz")
    else:
        channel_label.config(text="Recommended Channel: None")

    ax.clear()

    ax.plot(freq, power)

    ax.set_title("Real Time Spectrum Analyzer")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")

    if channel is not None:
        ax.axvline(channel, linestyle="--")


ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)

plt.show(block=False)

root.mainloop()