# DST Plate Reverb

This project implements a mono-in, stereo-out plate-style reverb (inspired by Jon Dattorro's topology) in a Jupyter notebook.

## Topology

- Topology image: [Reverb Topology](Reverb%20Topology.png)

![Reverb Topology](Reverb%20Topology.png)

## What Is Implemented

- `DelayLine`
  - Fixed-length circular buffer
  - `write(sample)` and `tap(n)` access (`n` samples behind write pointer)
- `OnePoleLowpassFilter`
  - One-pole lowpass with `bandwidth` or `damping` mapping
- `ModulatedAllpass`
  - Delay-based allpass with LFO-modulated delay time and allpass interpolation

The notebook wires these blocks into:

- Input diffusion allpasses (`APF1` to `APF4`)
- Two cross-coupled reverb tanks
- Modulated tank allpasses (`MAPF1`, `MAPF2`)
- Tank delays and additional allpasses (`Delay1..Delay4`, `APF5`, `APF6`)
- Stereo output taps (`y_L`, `y_R`) from specific points in the tank delay lines

## Files

- `PlateReverb.ipynb`: Main implementation and demo
- `snare.wav`: Input test signal
- `Reverb Topology.png`: Block diagram reference
- `dattorro.pdf`: Reference material

## Quick Start

1. Open `PlateReverb.ipynb`.
2. Run all cells from top to bottom.
3. The final cell:
   - loads `snare.wav`
   - processes it through `process_reverb_sample(...)`
   - displays dry/wet audio players
   - shows before/after spectrograms

## Python Dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `IPython` (for `Audio` display in notebook)

## Notes

- Reverb processing is sample-by-sample to match delay-network behavior.
- A short silent tail is appended so decay is audible/visible.