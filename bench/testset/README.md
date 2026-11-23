# Everyday test set

`sentences.json`: 40 sentences (8 imperative "command" sentences, 12 "you can ..." instruction sentences, 20 plain everyday sentences) taken from
**Simple English Wikipedia** and its help pages (articles on tea, bread, bicycle, weather, telephone, clock, kitchen, garden, and the editing and search help pages).
Text licence: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), attribution: Simple English Wikipedia contributors.
Chosen so that there are no names, digits or symbols: 6 to 18 words, letters, spaces and basic punctuation only, no capital letter after the first word (no proper names).

`make_testset.py` reads them with the heart voice and writes `clips16/` (needs a running `heart-server` and ffmpeg). The benchmark scripts build 94 test items from them:
the 17 single sentences of 4,500 ms or longer, plus every adjacent pair and triple joined with 400 ms of silence, up to 18,500 ms (2,385 reference words in all).
