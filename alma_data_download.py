from astroquery.alma import Alma
from astropy.io import fits
import os
from astropy.table import Table

alma = Alma()

alma.login("danielhill", store_password=True)

alma.archive_url = 'https://almascience.eso.org'
# llamatab = Table.read('/data/c3040163/llama/llama_main_properties.fits',format='fits')
llamatab = Table.read('/Users/administrator/Astro/LLAMA/llama_main_properties.fits',format='fits')


# llamatab.show_in_browser()

old_uids = ['uid://A001/X1465/X2b6', 'uid://A001/X1465/X28c', 'uid://A001/X1465/X2ba', 'uid://A001/X1284/Xdae', 'uid://A001/X1465/X2be', 'uid://A001/X1465/X2c4', 'uid://A001/X1296/X8f3', 'uid://A001/X1465/X290', 'uid://A001/X1465/X296', 'uid://A001/X1465/X286', 'uid://A001/X1465/X29c', 'uid://A001/X87d/X304', 'uid://A001/X1296/X8ef', 'uid://A001/X2fe/X665', 'uid://A001/X1465/X2a0', 'uid://A001/X1465/X2a6','uid://A001/X1465/X2ac','uid://A001/X1465/X2ce', 'uid://A001/X1465/X2b2', 'uid://A001/X1465/X2ca', 'uid://A001/X1465/X26a', 'uid://A001/X133d/X279e','uid://A001/X1465/X26e','uid://A001/X121/X176', 'uid://A001/X1296/X8e7','uid://A001/X2fe/X671', 'uid://A001/X1465/X274', 'uid://A001/X1465/X278','uid://A001/X1296/X648','uid://A001/X133d/X8e7','uid://A001/X1465/X27c','uid://A002/X5d7935/X293','uid://A001/X87a/X499','uid://A001/X1465/X280']
# # Theres no ngc 1365 in here because new data on monday 2/2/25, 4388 skipped because better data 13/2/25, PHANGS objects ngc2775 and ngc3351 and ngc4254 are skipped because they're probably good pics
# # NOTE: HOW TO DOWNLOAD MORE THAN ONE UID FOR A SINGLE OBJECT? FOR EXAMPLE: 7172 3783 4235
# for i in range(len(llamatab)):
#     os.system(f'cd /data/c3040163/raw/')
#     name = llamatab['name'][i]
#     if name != 'IC4653' or 'NGC 1365' or 'NGC 2775' or 'NGC 3351' or 'NGC 4254' or 'NGC 4388':
#         id = llamatab['id'][i]
#         print(name)
#         os.system(f'mkdir /data/c3040163/raw/{id}')
#         os.system(f'cd /data/c3040163/raw/{id}')
#         alma.retrieve_data_from_uid(uids[i],cache=True)
#         print('data downloaded')



############ uids for additional data ###################

############ How to have it not in explicit LLAMAtab order? ###############

#alma.help_tap()

def get_array_type(antenna_string):
    antennas = antenna_string.split()
    has_12m = any('DV' in ant for ant in antennas)
    has_7m = any('CM' in ant or 'DA' in ant for ant in antennas)
    has_tp = any('PM' in ant for ant in antennas)

    if has_12m:
        return '12m'
    elif has_7m:
        return '7m ACA'
    elif has_tp:
        return 'TP'
    else:
        return 'Unknown'

# for uid in uids:
#     print(f"Querying ivoa.obscore for Member OUS UID: {uid}")
#     query = f"""
#     SELECT * 
#     FROM ivoa.obscore
#     WHERE member_ous_uid = '{uid}'
#     """
#     try:
#         data = alma.query_tap(query).to_table()
#         if len(data) == 0:
#             print(f"No results found for UID {uid}")
#             continue

#     except Exception as e:
#         print(f"Error querying UID {uid}: {e}")
    
#     science_rows = data[data['science_observation'] == 'T']
#     science_target = science_rows['target_name'][0]
#     print(science_target)
#     array_type = get_array_type(science_rows['antenna_arrays'][0])
#     print(array_type)
#     names = llamatab['name'].tolist()
#     tabindex = names.index(science_target)
#     id = llamatab['id'][tabindex]
#     os.system(f'cd /data/c3040163/raw/{id}')
#     dirname = array_type
#     count = 1
#     while os.path.exists(dirname):
#         count += 1
#         dirname += f'_{count}'
#     os.makedirs(dirname)
#     os.system(f'cd /data/c3040163/raw/{id}/{dirname}')
#     alma.retrieve_data_from_uid(uid,cache=True)
        
import re

def is_line_covered(uid, obs_freq_hz):
    query = f"""
    SELECT frequency_support
    FROM ivoa.obscore 
    WHERE member_ous_uid = '{uid}'
    """
    try:
        result = alma.query_tap(query).to_table()
    except Exception as e:
        print(f"Query error for {uid}: {e}")
        return False

    if len(result) == 0 or result[0]['frequency_support'] in (None, ''):
        print(f"No frequency support info for {uid}")
        return False

    raw_support = result[0]['frequency_support']
    pattern = re.compile(r'(\d+\.\d+)\.\.(\d+\.\d+)GHz')
    matches = pattern.findall(raw_support)

    if not matches:
        print(f"No frequency ranges found in support string for {uid}")
        return False

    for f_min_str, f_max_str in matches:
        f_min = float(f_min_str) * 1e9
        f_max = float(f_max_str) * 1e9
        #print(f"Checking {f_min/1e9:.2f}–{f_max/1e9:.2f} GHz against target {obs_freq_hz/1e9:.2f} GHz")
        if f_min <= obs_freq_hz <= f_max:
            print(f"✅ Match found for UID {uid}")
            return True

    print(f"❌ No match found for UID {uid}")
    return False




line_freq = 230.538e9

for name in llamatab['name']:
    if name == 'IC4653' or 'NGC 1365' or 'NGC 2775' or 'NGC 3351' or 'NGC 4254' or 'NGC 4388':
        continue
    print(name)

    table_index = llamatab['name'].tolist().index(name)
    z = llamatab['redshift'][table_index]
    id = llamatab['id'][table_index]

    data = alma.query_object(name)
    uids = list(set(data['member_ous_uid']))
    for uid in uids:
        if uid in old_uids:
            continue
        if is_line_covered(uid, line_freq/(1+z)):
            # Filter to current UID
            science_rows = data[(data['member_ous_uid'] == uid) & (data['science_observation'] == 'T')]
            
            if len(science_rows) == 0:
                print(f"No science observation for {uid}")
                continue
            array_type = get_array_type(science_rows['antenna_arrays'][0])
            print(f" {array_type}")


