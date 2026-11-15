# Labelling guide

The AI learns how much each terrain measurement should count by studying pairs
of places that people have marked as alike or different. These labels are the
only thing it learns from, so their quality decides how good it is.

## Make the sheet

```powershell
python make_label_sheet.py
```

This writes `labels/label_sheet.csv`. Each row pairs the Mars target near
Jezero crater with one Earth tile. The sheet doesn't show the app's scores, so
the labels come from your own judgement rather than the program's.

## Label a pair

1. Look at the Mars target in the web app (the left picture at the top).
2. Open the row's `satellite_view` link to see the Earth tile from above.
3. Fill in the `label` column:
   - `1` if the two places show the same kind of landform at this 10 km scale,
     for example fan or delta deposits, channels, or layered cliffs
   - `0` if they clearly differ, for example a dune sea compared with a rocky plain
   - leave it blank if you're not sure
4. Write a short `reason` and your initials in `labeller`.

The `catalogue_mars_features` column says which Mars features a published
catalogue lists for a nearby Earth site. Use it as a hint, not the answer: a
site can be a good analogue for one feature without the whole 10 km square
looking like Jezero.

Don't look at the app's scores while labelling.

## How many

The training tool needs at least 10 pairs marked `1` and 10 marked `0`. Aim
for 30 or more of each. Ask a teacher or mentor to check some rows, for
example every fifth one, and note where you disagree.

If you edit the sheet in Excel, save it with **Save As > CSV UTF-8 (Comma
delimited)** and keep the column names as they are.

## Train

```powershell
python train_model.py
```

It prints how well the learned weights sort pairs it didn't train on, next to
the app's current hand-set weights, and saves `models/learned_weights.json`.
With fewer than about 100 labels, treat the numbers as rough.

The catalogue sites come from the Catalogue of Planetary Analogues (Preston,
Grady and Barber, The Open University, 2012, ESA contract 4000104716/11/NL/AF).
