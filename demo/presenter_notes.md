# TAORG demo: 30-second explanation

*(About 85 words. Start with the method set to EMA, then switch to GatedEMA at the occlusion.)*

This is a synthetic scenario with controlled observations; nothing is detected from video.
On the left, a person holds a cup. In the middle is what a detector reports frame by frame.
On the right is what our temporal methods output from exactly those reports.

When the board hides the cup, the detector loses "holding". No Memory drops it immediately.
EMA keeps it briefly, then loses it. The visibility-aware filter keeps it, because the input says
the cup is occluded.

When the cup is really put down, every method removes the relation. EMA is slower: about a third
of a second. These are illustrative settings, not a result.
