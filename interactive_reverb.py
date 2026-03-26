import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    import numpy as np
    import math
    import scipy.io.wavfile as wavfile
    from scipy.signal import spectrogram as scipy_spectrogram
    import matplotlib.pyplot as plt
    import io
    return io, math, mo, np, plt, scipy_spectrogram, wavfile

@app.cell
def __(mo):
    mo.md(
        r"""
        # Plate Reverb Interactive Demo
        This is an interactive demonstration of a typical plate reverb topology.
        Use the controls below to configure the reverb, and click **Process Reverb** to hear the results!
        """
    )
    return

@app.cell
def __(mo):
    image_viewer = mo.image(src="Reverb Topology.png", rounded=True)
    mo.vstack([
        mo.md("### Topology Reference"),
        image_viewer
    ])
    return image_viewer,

@app.cell
def __(mo):
    controls = mo.ui.dictionary({
        "audio_file": mo.ui.file(filetypes=[".wav"], kind="area", label="Custom Audio File (.wav) - optional"),
        "decay": mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Decay Time"),
        "pre_delay_ms": mo.ui.slider(start=0.0, stop=100.0, step=1.0, value=20.0, label="Pre-delay (ms)"),
        "lpf_damping": mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Lowpass Filter Smoothing"),
    })
    form = mo.ui.form(element=controls, label="Process Reverb", show_clear_button=True)
    
    mo.vstack([
        mo.md("### Parameters"),
        form
    ])
    return controls, form

@app.cell
def __(math):
    class DelayLine:
        def __init__(self, lengthSamples):
            self.length = int(lengthSamples)
            self.buffer = [0.0] * self.length
            self.writeIdx = -1

        def setDelay(self, lengthSamples):
            self.length = int(lengthSamples)
            if self.length > len(self.buffer):
                self.buffer.extend([0.0] * (self.length - len(self.buffer)))
            elif self.length < 1:
                self.length = 1

        def tap(self, n):
            n = int(n)
            if n < 0: n = 0
            if n >= self.length: n = self.length - 1
            readIdx = (self.writeIdx - n) % self.length
            return self.buffer[readIdx]

        def write(self, x):
            self.writeIdx = (self.writeIdx + 1) % self.length
            self.buffer[self.writeIdx] = float(x)

        def process(self, x):
            y = self.tap(self.length - 1)
            self.write(x)
            return y

    class OnePoleLowpassFilter:
        def __init__(self, control=0.5, mode="damping", initialState=0.0):
            self.mode = str(mode)
            self.control = None
            self.a = 0.0
            self.z1 = float(initialState)
            self.setControl(control, mode)

        def setControl(self, control, mode=None):
            if mode is not None:
                self.mode = str(mode).lower()
            self.control = float(control)
            self.a = self.control if self.mode == "damping" else (1.0 - self.control)

        def process(self, x):
            x = float(x)
            y = (1.0 - self.a) * x + self.a * self.z1
            self.z1 = y
            return y

    class ModulatedAllpass:
        def __init__(self, maxDelaySamples, delaySamples, gain=0.5, sampleRate=48000.0, lfoRateHz=5.0, lfoDepthSamples=0.0, lfoPhase=0.0, fillValue=0.0):
            self.maxDelay = float(maxDelaySamples)
            self.delay = float(delaySamples)
            self.gain = float(gain)
            self.sampleRate = float(sampleRate)
            self.lfoRate = float(lfoRateHz)
            self.lfoDepth = float(lfoDepthSamples)
            self.lfoPhase = float(lfoPhase)
            self.buffer = [float(fillValue)] * (int(self.maxDelay) + 3)
            self.writeIdx = -1
            self.apX1 = 0.0
            self.apY1 = 0.0
            self.setDelay(delaySamples)

        def setDelay(self, delaySamples):
            d = max(0.0, min(float(delaySamples), self.maxDelay))
            self.delay = d

        def tap(self, n):
            d = max(0.0, min(float(n), self.maxDelay))
            nInt = int(d)
            frac = d - nInt
            x0 = self.buffer[(self.writeIdx - nInt) % len(self.buffer)]
            if frac < 1e-12:
                return x0
            eta = (1.0 - frac) / (1.0 + frac)
            y = eta * x0 + self.apX1 - eta * self.apY1
            self.apX1, self.apY1 = x0, y
            return y

        def process(self, x):
            modDelay = self.delay + self.lfoDepth * math.sin(self.lfoPhase)
            d = self.tap(modDelay)
            w = float(x) + self.gain * d
            y = d - self.gain * w
            self.writeIdx = (self.writeIdx + 1) % len(self.buffer)
            self.buffer[self.writeIdx] = w
            return y

    return DelayLine, ModulatedAllpass, OnePoleLowpassFilter

@app.cell
def __(np, plt, scipy_spectrogram):
    def generate_spectrogram(x, Fs, title='Log-Frequency Spectrogram', NFFT=4096, noverlap=3584):
        f, t, Sxx = scipy_spectrogram(x, fs=Fs, window='hann', nperseg=NFFT, noverlap=noverlap, nfft=8192, scaling='spectrum')
        Sxx_db = 10 * np.log10(Sxx + 1e-20)
        Sxx_db -= np.max(Sxx_db)
        Sxx_db = np.clip(Sxx_db, -60, 0)

        f_log = np.logspace(np.log10(max(f[1], 1)), np.log10(f[-1]), len(f))
        Sxx_log = np.zeros((len(f_log), Sxx_db.shape[1]))
        for i in range(Sxx_db.shape[1]):
            Sxx_log[:, i] = np.interp(f_log, f, Sxx_db[:, i])

        fig, ax = plt.subplots(figsize=(10, 4))
        mesh = ax.pcolormesh(t, f_log, Sxx_log, shading='gouraud', cmap='viridis')
        ax.set_yscale('log')
        ax.set_ylim(f_log[0], f_log[-1])
        ticks = np.unique(np.logspace(np.log10(f_log[0]), np.log10(f_log[-1]), 10, dtype=int))
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(tk) for tk in ticks])
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (Hz)')
        fig.colorbar(mesh, ax=ax, label='Intensity (dB)')
        # Important to tighten layout
        fig.tight_layout()
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return mo.image(src=buf)
    return generate_spectrogram,

@app.cell
def __(
    DelayLine,
    ModulatedAllpass,
    OnePoleLowpassFilter,
    form,
    generate_spectrogram,
    io,
    mo,
    np,
    plt,
    wavfile
):
    if not form.value:
        mo.stop(True, mo.md("**Adjust settings and click 'Process Reverb' to process the audio.**"))
        
    v = form.value
    decay_val = v["decay"]
    pre_delay_ms_val = v["pre_delay_ms"]
    lpf_val = v["lpf_damping"]
    
    import warnings
    warnings.filterwarnings('ignore', category=wavfile.WavFileWarning)
    Fs = 48000
    audio_data = None
    if v["audio_file"]:
        try:
            file_content = v["audio_file"][0].contents
            Fs, x = wavfile.read(io.BytesIO(file_content))
        except Exception:
            Fs, x = wavfile.read('snare.wav')
    else:
        try:
            Fs, x = wavfile.read('snare.wav')
        except Exception:
            Fs = 48000
            x = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000)).astype(np.float32)
            
    if len(x.shape) > 1:
        x = x.mean(axis=1)
        
    # Standardize scale
    if x.dtype != np.float32 and x.dtype != np.float64:
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
            
    # Subsample to a short snippet if too long for demonstration to avoid blocking UI for minutes
    max_duration = 5.0 # seconds
    if len(x) > int(max_duration * Fs):
        x = x[:int(max_duration * Fs)]
        
    with mo.status.spinner("Applying Plate Reverb (this may take a moment)..."):
        # Add tail
        input_audio = np.concatenate([x, np.zeros(int(4.0 * Fs), dtype=np.float32)])
        output_audio = np.zeros((len(input_audio), 2), dtype=np.float32)

        preDelay = DelayLine(max(1, int((pre_delay_ms_val / 1000.0) * Fs))) 
        lpf1 = OnePoleLowpassFilter(lpf_val, "bandwidth", 0.0)

        apf1 = ModulatedAllpass(maxDelaySamples=210, delaySamples=210, gain=0.75)
        apf2 = ModulatedAllpass(maxDelaySamples=158, delaySamples=158, gain=0.75)
        apf3 = ModulatedAllpass(maxDelaySamples=561, delaySamples=561, gain=0.625)
        apf4 = ModulatedAllpass(maxDelaySamples=410, delaySamples=410, gain=0.625)

        mapf1 = ModulatedAllpass(maxDelaySamples=1355, delaySamples=1343, gain=0.7, lfoRateHz=0.10, lfoDepthSamples=12)
        delay1 = DelayLine(6241)
        lpf2 = OnePoleLowpassFilter(0.5, "damping", 0.0)
        apf5 = ModulatedAllpass(maxDelaySamples=3931, delaySamples=3931, gain=0.5)
        delay2 = DelayLine(4681)

        mapf2 = ModulatedAllpass(maxDelaySamples=1007, delaySamples=995, gain=0.7, lfoRateHz=0.07, lfoDepthSamples=12)
        delay3 = DelayLine(6590)
        lpf3 = OnePoleLowpassFilter(0.5, "damping", 0.0)
        apf6 = ModulatedAllpass(maxDelaySamples=2664, delaySamples=2664, gain=0.5)
        delay4 = DelayLine(5505)

        leftTankFeedback = 0.0
        rightTankFeedback = 0.0

        for i, sample in enumerate(input_audio):
            preDelay.write(sample)
            u = preDelay.tap(preDelay.length - 1)
            u = lpf1.process(u)

            u = apf1.process(u)
            u = apf2.process(u)
            u = apf3.process(u)
            u = apf4.process(u)

            inL = u + decay_val * rightTankFeedback
            inR = u + decay_val * leftTankFeedback

            sL = mapf1.process(inL)
            sL = delay1.process(sL)
            sL = lpf2.process(sL)
            sL = apf5.process(sL)
            leftTankFeedback = delay2.process(sL)

            sR = mapf2.process(inR)
            sR = delay3.process(sR)
            sR = lpf3.process(sR)
            sR = apf6.process(sR)
            rightTankFeedback = delay4.process(sR) 

            yL = (delay1.tap(394) + delay1.tap(4401) - apf5.tap(2831) + delay2.tap(2954)
                  - delay3.tap(2945) - apf6.tap(277) - delay4.tap(1066))

            yR = (delay3.tap(522) + delay3.tap(5368) - apf6.tap(1817) + delay4.tap(3956)
                  - delay1.tap(3124) - apf5.tap(496) - delay2.tap(179))

            output_audio[i, 0] = yL
            output_audio[i, 1] = yR
            
    with mo.status.spinner("Generating Spectrograms..."):
        fig_before = generate_spectrogram(input_audio, Fs, 'Original Audio')
        fig_after = generate_spectrogram(output_audio.mean(axis=1), Fs, 'Reverberated Audio')
    
    # Render audio buffers
    buf_in = io.BytesIO()
    max_amp_in = np.max(np.abs(input_audio))
    norm_in = input_audio / max_amp_in if max_amp_in > 1.0 else input_audio
    wavfile.write(buf_in, Fs, (norm_in * 32767).astype(np.int16))
    buf_in.seek(0)
    audio_orig = mo.audio(buf_in)
    
    buf_out = io.BytesIO()
    
    # Normalize to prevent integer overflow wrapping
    max_amp = np.max(np.abs(output_audio))
    norm_audio = output_audio / max_amp if max_amp > 1.0 else output_audio
        
    wavfile.write(buf_out, Fs, (norm_audio * 32767).astype(np.int16))
    buf_out.seek(0)
    audio_rev = mo.audio(buf_out)

    result_ui = mo.vstack([
        mo.md("### Results"),
        mo.hstack([
            mo.vstack([
                mo.md("**Original Audio**"),
                audio_orig,
                fig_before
            ]),
            mo.vstack([
                mo.md("**Reverberated Audio**"),
                audio_rev,
                fig_after
            ])
        ])
    ])
    
    result_ui

    return (
        audio_orig,
        audio_rev,
        buf_in,
        buf_out,
        decay_val,
        fig_after,
        fig_before,
        lpf_val,
        pre_delay_ms_val,
        result_ui,
        v,
    )

if __name__ == "__main__":
    app.run()