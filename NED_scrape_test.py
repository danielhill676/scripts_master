from astroquery.ipac.ned import Ned
result_table = Ned.query_object("MCG-05-14-012")
#print(result_table)
print(result_table['RA'][0]+123.5345) # an astropy.table.Table

