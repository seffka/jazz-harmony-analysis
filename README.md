# jazz-harmony-analysis
This repository contains accomponying code for my master thesis "Automatic Harmony Analysis of Jazz Audio Recordings".

To run tools the following environment variables must be set:
   * JAZZ_HARMONY_DATA_ROOT path to base directory with audio data which is refered from
     annotations.
   * JAZZ_HARMONY_CACHE_DIR path to the directory where cache (e.g. with chroma features
     and beats) will reside.

Dependencies:
   * vamp PyPi package
   * NNLS chroma plugin
   * essentia
   * musicbrainzngs: https://github.com/alastair/python-musicbrainzngs
   * MusOOEvaluator (MTG fork): https://github.com/MTG/MusOOEvaluator.git
     NOTE: don't forget to "git submodule update --init --recursive"
   * madmom: https://github.com/CPJKU/madmom
   * for sonfication MATLAB is needed and the following toolboxes:
      * Loudness toolbox (http://genesis-acoustics.com/en/loudness_online-32.html)
      * MIDI toolbox 1.1 (https://github.com/miditoolbox/1.1)

for sonification, it is needed to:
   * run matlab with 'matlab' command, so it should be in your PATH (or aliased).
   * utils/MATLAB directory should be added to matlab pathes.
