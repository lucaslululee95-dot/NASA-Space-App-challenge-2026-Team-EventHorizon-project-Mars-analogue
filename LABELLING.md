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

The easiest way is the labelling page, which runs on your own PC:

```powershell
python -m streamlit run label_app.py
```

It shows the Mars target and one Earth tile side by side, as shaded relief and as
a satellite image, both 10 km across. Type your initials in the sidebar, then for
each pair:

- click **Alike (1)** if they show the same kind of land at this scale, for example
  a gently rolling rocky plain like the Mars picture
- click **Different (0)** if they clearly differ, for example steep mountains, a sea
  of large dunes, or a perfectly flat salt pan
- click **Not sure, skip** if you can't tell

Add a few words in **Reason** before you click. Each answer is saved to
`labels/label_sheet.csv` straight away, and the page moves to the next unlabelled
pair. The **See the Earth place in Google Maps** button shows more detail.

You can also edit the CSV in a spreadsheet: fill in the `label`, `reason` and
`labeller` columns. Close the spreadsheet before using the labelling page, or it
can't save.

The hint about a catalogue site says which Mars features a published
catalogue lists for a nearby Earth site. Use it as a hint, not the answer: a
site can be a good analogue for one feature without the whole 10 km square
looking like Jezero.

Don't look at the app's scores while labelling.

## How many

The training tool needs at least 10 pairs marked `1` and 10 marked `0`. Aim
for 30 or more of each. Ask a teacher or mentor to check some rows, for
example every fifth one, and note where you disagree.

If you edit the sheet in a spreadsheet, save it as **CSV UTF-8** and keep the
column names as they are.

## Train

```powershell
python train_model.py
```

It prints how well the learned weights sort pairs it didn't train on, next to
the app's current hand-set weights, and saves `models/learned_weights.json`.
With fewer than about 100 labels, treat the numbers as rough.

The catalogue sites come from the Catalogue of Planetary Analogues (Preston,
Grady and Barber, The Open University, 2012, ESA contract 4000104716/11/NL/AF).
