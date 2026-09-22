"""Explicit, reviewable reference admission; these are not expert quality scores."""

# Exact references appearing in CoFID 2021 and its official user-guide
# bibliography. Food-level references do not prove every constituent was
# analysed. They may support curated TRAINING only, never strict validation.
COFID_ANALYTICAL_REFERENCES = frozenset({
    "DH, Nutrient analysis of fish and fish products, 2013",
    "DH, Nutrient analysis of fruit and vegetables, 2013",
    "PHE, Nutrient analysis of fresh and processed fruit and vegetables with respect to fibre, 2017",
    "LGC, Nutrient analysis of retail cuts of lamb, 1993-1994",
    "LGC, Nutrient analysis of carcase beef, 1992-1993",
    "DH, Nutrient analysis of biscuits, buns, cakes and pastries, 2011",
    "DH, Nutrient analysis of a range of processed foods with particular reference to trans fatty acids, 2013",
    "LGC, Nutrient analysis of chicken and turkey, 1994-1995",
    "LGC, Nutrient survey of flours and grains, 2005",
    "QIB, Report on nutrient analysis of key cuts of pork, 2020",
    "LGC, Nutrient analysis of retail cuts of pork, 1992-1993",
    "LGC, Fruit and vegetables, 1989-1990",
    "Direct Laboratories, Nutrient analysis catch up project, 2003",
    "University of Leeds, Nutritional analysis of commonly consumed South Asian foods in the UK, 2007",
    "LGC, Nutritional survey fish and fish products, 1986-1987",
    "IFR, The nutritional composition of retail vegetables in the UK, 1984-1987",
    "LGC, Nutrient analysis of pasta and pasta sauces, 2004",
    "LGC, Nutritional composition of fresh fruit, 1985-1986",
    "DH, Nutrient analysis of a range of processed foods with particular reference to trans fatty acids, 2011",
    "LGC, Analytical survey of fish and fish products, 1991-1992",
    "LGC, Nutrient analysis of cheese, 1999",
})

# Published Norwegian analytical projects, identified from the official
# sources endpoint. Imports from other FCDBs and generic industry data are
# intentionally absent. Source descriptions and URLs are saved in staging.
NORWAY_ANALYTICAL_REFERENCES = frozenset({
    "216", "222a", "224a", "225a", "227a", "228a", "231", "232",
    "233", "234", "235", "237", "330", "332", "334", "336",
})
