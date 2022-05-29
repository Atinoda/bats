# Binary Applied Threshold Segmentation

This time series segmentation algorithm was developed for the ICE Overlay Digitalisation system.

## Overview
Binary Action Threshold Segmentation (BATS) has been developed to split univariate time series into a sequence of ON and OFF states.The algorithm achieves this by classifying each individual data point as a binary state then running a sequence of filters across the resulting classifications to reject interruptions and to unify temporal regions of the same state.

The algorithm outputs a list of segments, their operating points, and their indices - or alternatively, can be considered as identifying the hinge points within a time series. BATS is designed to work with unevenly spaced time series without compromise (ie. it is robust to missing data and irregularly spaced measurements).

BATS is most suitably applied to time series data which can be well approximated by a fixed amplitude square wave, however it can also produce good results for a process which is approximated by a varying amplitude square wave.

## Publications
Details are published at: [x]

If you use this algorithm in your work, please cite as: [x]
