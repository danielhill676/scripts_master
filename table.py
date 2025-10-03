from astropy.table import Table
llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits',format='fits')

# llamatab.pprint_all()
for i in range(len(llamatab)):
    if llamatab['name'][i] == 'NGC 1315':
        print(llamatab['redshift'][i])