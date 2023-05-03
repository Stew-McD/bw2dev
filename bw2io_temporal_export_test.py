
"""
Description: A fix for importing and exporting databases that have exchanges with temporal distributions
Author: Stewart Charles McDowall
Github: SC-McD
Email: s.c.mcdowall@cml.leidenuniv.nl
Date: 2023-05-03

### Testing script (without editing bw2io source code)
#### [TD_ImportExport_Fix.py]()
#### Steps 
1. creates a new project and database, adds two activities and two exchanges
2. exports the database to excel (this works)
3. deletes the database and imports the database from excel (this works)
4. adds temporal distributions to the exchanges
5. tries to export the database to excel (this fails)
6. converts the temporal distributions to (three) strings stored in the exchange dictionary and exports the database to excel (this works)
7. deletes the database and imports the database from excel (this works)
8. converts the three strings back into their original form as a temporal distribution, deletes the strings from the exchange dictionary

"""

"""
### Problem
When temporal distributions are added to exchanges, the database cannot be exported to excel/csv.
This is because the temporal distributions are stored as numpy arrays.

### Solution
#### function 1: TD_to_string()
Converts all temporal distributions in a database to a set of strings (using np.array2string)
TD (bwt object)--> TD_amount, TD_time, TD_time_type (strings)  
#### function 2: string_to_TD()
Converts the strings back into their original form as TD objects (using np.fromstring) (after importing the database from excel) 

### Possible implementation of the solution in to bw2io (optimal is a different story)
* Add the two new functions to bw2io.strategies
* Call TD_to_string() in bw2io.export.csv.CSVFormatter.get_formatted_data
* Call string_to_TD() in bw2io.importers.excel.ExcelImporter



"""
#%%
import bw2data as bd
import bw2io as bi
import bw_temporalis as bwt
import numpy as np



print('bw2io:', bi.__version__)
print('bw2data:', bd.__version__)
print('bw_temporalis:', bwt.__version__)

# %%  Create new project and database

project_name = "test_project_TD_IO"
db_name = "test_db"
bio_name = "biosphere"

bd.projects.set_current(project_name)
bd.projects.delete_project(project_name, delete_dir=True)
bd.projects.set_current(project_name)

db = bd.Database(db_name)
db.register()

bio = bd.Database("biosphere")
bio.register()

# Add activities and exchanges to database

act1 = db.new_node(
    name="test_name1",
    code="test_code1",
)
act1.save()

act2 = db.new_node(
    name="test_name2",
    code="test_code2",
)

# add exchanges

exc1 = act2.new_edge(
    input=act1,
    amount=10,
    type="technosphere",
)
exc1.save()
act2.save()
exc2 = act1.new_edge(
    input=act2,
    amount=100,
    type="technosphere",
)
exc2.save()

db.delete_duplicate_exchanges()
# %% Check database
for act in db:
    print("activity in database:")
    print("\t", act.as_dict())
    for exc in act.technosphere():
        print("technosphere exchanges in activty:")
        for k, v in exc.as_dict().items():
            print("\t", k, ":", v, ", type:", type(v))
    for exc_bio in act.biosphere():
        print("biosphere exchanges in database:")
        for k, v in exc_bio.as_dict().items():
            print("\t", k, ":", v, ", type:", type(v))
# %% Test export and import without temporal data

# export database to excel
try:
    path = bi.export.write_lci_excel(db_name)
    print("Database exported to:\n", path)
except Exception as e:
    print(e)

# delete existing database

del bd.databases["test_db"]
del db

# import database from excel
bi.create_core_migrations()
xl_importer = bi.importers.ExcelImporter(path)
xl_importer.apply_strategies()
xl_importer.match_database(fields=["name", "unit", "location"])
xl_importer.write_database()

# check imported database
db = bd.Database(db_name)
for act in db:
    print("activity in database:")
    print("\t", act.as_dict())
    for exc in act.technosphere():
        print("technosphere exchanges in activty:")
        for k, v in exc.as_dict().items():
            print("\t", k, ":", v, ", type:", type(v))
    for exc_bio in act.biosphere():
        print("biosphere exchanges in database:")
        for k, v in exc_bio.as_dict().items():
            print("\t", k, ":", v, ", type:", type(v))

# %% Add temporal distribution to exchange and try to save (fails)

exc1["TD"] = bwt.TemporalDistribution(
    np.array([0, 1, 2, 3, 4], dtype="timedelta64[Y]"),
    np.array([2.43, 3.32, 34, 123, 4]),
)
exc1.save()


# a different way to add temporal data to an exchange
length = 100
date = np.linspace(-50, 50, length, dtype="timedelta64[Y]", endpoint=True)
amount = np.around(np.random.uniform(low=0.0, high=200.0, size=100), decimals=2)
exc2["TD"] = bwt.TemporalDistribution(date, amount)
exc2.save()

db.delete_duplicate_exchanges()

#%% Test export and import with temporal data

try:
    path = bi.export.write_lci_excel(db_name)
    print(path)
except Exception as e:
    print("Export failed:", e)

# "Export failed: expected string or bytes-like object, got 'TemporalDistribution'"
# %% Convert temporal data to string and try to save

# define function to convert temporal data to strings
def TD_to_string(db):
    for act in db:
        for exc in act.technosphere():
            if "TD" in exc:
                exc['TD_amount'] = np.array2string(exc['TD'].amount, 
                    separator=", ").replace('[', "").replace(']',"").replace("\n", "")
                exc['TD_time'] = np.array2string(exc['TD'].date, 
                    separator=", ").replace('[', "").replace(']',"").replace("\n", "")
                exc['TD_time_type'] = str(exc['TD'].base_time_type)
                exc.pop("TD")
                exc.save()
                print(exc.as_dict())
                print("converted temporal data to strings")

# export database to excel with temporal data as strings

TD_to_string(db)

try:
    path = bi.export.write_lci_excel(db_name)
    print("exported", db.name, "to:", path)
except Exception as e:
    print("Export failed:", e)

#%% import database from excel with temporal data as strings

# delete existing database

del bd.databases["test_db"]
del db

# import database from excel
#bi.create_core_migrations()
xl_importer = bi.importers.ExcelImporter(path)
xl_importer.apply_strategies()
xl_importer.match_database(fields=["name", "unit", "location"])
xl_importer.write_database()

# check imported database
db = bd.Database(db_name)
for act in db:
    print("activity in database:")
    print("\t", act.as_dict())
    for exc in act.technosphere():
        print("technosphere exchanges in activty:")
        for k, v in exc.as_dict().items():
            print("\t", k, ":", v, ", type:", type(v))
    for exc_bio in act.biosphere():
        print("biosphere exchanges in database:")
        for k, v in exc_bio.as_dict().items():
            print("\t", k, ":", v, ", type:", type(v))
    

#%% convert temporal data to TD object

def string_to_TD(db):

    for act in db:
        for exc in act.technosphere():
            try:
                td_amount = np.fromstring(exc['TD_amount'], sep=', ', dtype=float)
                td_date = np.fromstring(exc['TD_time'], sep=', ', dtype=exc['TD_time_type'])
                exc["TD"] = bwt.TemporalDistribution(td_date, td_amount)
                
                exc.pop("TD_amount")
                exc.pop("TD_time")
                exc.pop("TD_time_type")
                exc.save()
                print("converted strings to temporal data")

            except KeyError:
                print("no temporal distribution")

#string_to_TD(db)

# check imported database
db = bd.Database(db_name)
for act in db:
    print("activity in database:")
    print("\t", act.as_dict())
    for exc in act.technosphere():
        print("technosphere exchanges in activty:")
        for k, v in exc.as_dict().items():
            print("\t", k, ":", v, ", type:", type(v))
    for exc_bio in act.biosphere():
        print("biosphere exchanges in database:")
        for k, v in exc_bio.as_dict().items():
            print("\t", k, ":", v, ", type:", type(v))

#%% 

